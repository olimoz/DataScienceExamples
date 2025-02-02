#%%
### CRITIC FOR THE AUTOENCODER

# This is a quick experiement I wanted to try out...not a major project.

# When a human looks at reconstructions of records made by an autoencoder, or
# at fecords generated by a VAE, they will likely critique the record
# This would be done by asking whether one figuire is out of context compared with 
# the others. An accountant may ask whether a figure is even possible in the context of the others
# The autoencoder is considered the 'first' moodel.

# The second model is new and it attempts to build a critic.
# There is a mini encoder for each field in the input data.
# That encoder is fed by all other fields in the sequence, but NOT the one it seeks to generate.
# In effect, the model is being asked to generate each field given all other fields, but not the field itself
# If a value in a field does not belong to the training distribution then this new model should generate a notably different value

# A third model is then constructed.
# The third model has access to
#   a) the reconstruction presented by the autoencoder
#   b) the latent variables (2dimensions) used to build the reconstruction
#   c) the critic's opinion (output) of the reconstruction
# The third model does not have access to the original input to the autoencoder
# This third model is trained to use its inputs in order to refine the reconstruction

# RESULTS

# The result is that reconstruction errors improve and the distance between 
# mean errors on fake data vs real data increases, making the tool more reliable 
# for identifying how fake a record is.
#   Mean reconstruction errors on test data (previously unseen):
#                 Real Data    Fake Data
#   Autoencoder:  0.3529       0.9435
#   Mixed Model:  0.3381       1.0635

# These results are from 500 batches of 32 records per batch. Significance analysis (t-test) pending
# The third model makes a small improvement to the reconstructions of real data, but
# substantially reduces the ability to reproduce fake data, which is good.
# This increases the reconstruction error distance between real and fake data, 
# thus improving its usefulness for scoring other generative models on their 'fakeness'

#%%
### NOTE ON CUSTOM LAYERS IN KERAS

# This model uses custom layers, lambda layers which call tf.boolean_mask and apply various masks
# The objective is to mask the current input field from each layer, but permit other fields
# Thus there is a different boolean mask for each input field. 
# This appears to cause all sorts of curious problems with Keras.
# The model works fine when defined outside of a function. When defined inside a function
# then the final field's mask applies itself across all layers, which is strange as it does not happen when defined outside a function
# Furthermore, the keras model fails to load correctly from file, even when custom_objects are defined
# It becomes necessary to define the model in code and then load saved weights.

# The issues may derive from how keras and Tensorflow define scope for operators and variables. tf.boolean_mask is an operator.

# It may be better to import keras layers from tf.keras instead of from Keras.
# Thus work entirely within the tensorflow implementation of keras.


#%%
import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from os import listdir, chdir
from os.path import isfile, isdir, join
import datetime
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization, concatenate, Activation
from keras.losses import mse, binary_crossentropy
from keras.models import Model,load_model
from keras import metrics
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf

proj_root = 'E:\Data\CompaniesHouse'

#%%
# LOAD DATA

prepped_10cols = pd.read_csv(os.path.join(proj_root, 'prepped_10cols.csv'))
sds_10cols     = pd.read_csv(os.path.join(proj_root, 'sds_10cols.csv'), index_col=0)
means_10cols   = pd.read_csv(os.path.join(proj_root, 'means_10cols.csv'), index_col=0)

x_train_10cols = pd.read_csv(os.path.join(proj_root, 'x_train_10cols.csv'), index_col='Report_Ref')
x_train_10cols = x_train_10cols.drop(columns=['Unnamed: 0'])

x_valid_10cols = pd.read_csv(os.path.join(proj_root, 'x_valid_10cols.csv'), index_col='Report_Ref')
x_valid_10cols = x_valid_10cols.drop(columns=['Unnamed: 0'])

x_testi_10cols = pd.read_csv(os.path.join(proj_root, 'x_testi_10cols.csv'), index_col='Report_Ref')
x_testi_10cols = x_testi_10cols.drop(columns=['Unnamed: 0'])

# confirm we are running on GPU, not CPU
from tensorflow.python.client import device_lib
print([(item.name, item.physical_device_desc) for item in device_lib.list_local_devices()])

#%%
masks = []
for current_field in range(12):
    mask = np.repeat(True, 22)
    # get mask excluding the current field
    mask[current_field] = False
    # first two items always true
    if current_field >= 2:
        mask[current_field+10] = False
    masks.append(mask)

# function to help with lambda layers
def apply_mask(tensor, mask):
    import tensorflow as tf # we do this so model works when loading from file
    tensor_masked = tf.boolean_mask(tensor, mask, axis=1) # axis=1 means apply mask to columns
    return tensor_masked

def define_and_compile( input_df,
                        encode_layer_nodes,
                        apply_mask          = apply_mask,
                        masks               = masks,
                        has_nan             = False,
                        apply_batchnorm     = False,
                        leaky_alpha         = 0.2,
                        kinit               = 'glorot_normal'):
    #Testing values
    #input_df            = x_train_10cols
    #encode_layer_nodes  = [16,8,4]
    #has_nan             = False
    #apply_batchnorm     = False
    #leaky_alpha         = 0.2
    #kinit               = 'glorot_normal'

    ## Instantiate inputs ##
    original_dim = len(input_df.columns)

    # Input to Encoder (from data to be modelled)
    inputs_encoder = Input(shape=(original_dim,), name='encoder_input')

    # Each input field gets its own encoder
    # That encoder takes all fields EXCEPT the current field
    # This slicing is done by a lambda layer. The encoder is simply dense layers
    # we store these pathways in a list
    encoder_paths = list(range(12))

    # for each regression value in the input (not binary field)
    for current_field in range(12):

        current_mask = np.asarray(masks[current_field])

        # also exclude current field's binary counterpart. 
        # NB each field, bar the first 2, has a binary counterpart which represents whether it is a +ve or -ve value
        if current_field >= 2:
            final_nodes = 2
        else:
            # first two columns. no binary partner so no additional masking of inputs
            # but we must ensure only one output for one input
            final_nodes = 1
        
        # use boolean mask to remove the current field's column of data
        encoder_paths[current_field] = Lambda(lambda tensor: apply_mask(tensor, current_mask), 
                                                mask=None, # must forget previous mask
                                                name='lambda_field_'+str(current_field),
                                                output_shape=[original_dim-final_nodes])(
                                                    inputs_encoder)

        # for each Lambda layer there is an encoder:
        for layer_nodes in encode_layer_nodes:
            
            # define layer
            encoder_paths[current_field] = Dense(
                            units = layer_nodes, 
                            name = 'dense_'+str(current_field)+'_'+str(layer_nodes),
                            kernel_initializer=kinit)(
                                encoder_paths[current_field])
                            
            # optionally apply batch norm
            if(apply_batchnorm):
                encoder_paths[current_field] = BatchNormalization()(encoder_paths[current_field])
            
            # define activation
            encoder_paths[current_field] = LeakyReLU(alpha= leaky_alpha, 
                                                        name = 'lkyrelu_'+str(current_field)+'_'+str(layer_nodes))(
                                                            encoder_paths[current_field])
        # END OF FOR LOOP

        # the final layer for each encoder has two nodes
        # one for the value and one for its binary partner, which indicates whether the value is +ve or -ve
        # No activation is applied at this stage, that comes after another lambda layer
        # The value eventually gets linear activiation. The binary gets sigmoid
        encoder_paths[current_field] = Dense(
                            units = final_nodes, 
                            name = 'dense_'+str(current_field)+'_final',
                            kernel_initializer=kinit)(
                                encoder_paths[current_field])

    # END OF FOR LOOP

    # concatenante outputs into one layer
    recons = concatenate(encoder_paths)

    # the reconstruction is in value-binary pairs, (values = regression values)
    # wheras we want all the values followed by all the binaries, just like the input
    # to achieve this we use another mask
    mask_regres = list(range(original_dim))

    # the regression values are to be the first item of each pair, ie the evens OR the first two
    mask_regres = np.array([True if item % 2 == 0 or item <= 2 else False for item in mask_regres])

    # the mask for the binaries is simply the opposite of the above
    mask_binary = np.array([not item for item in mask_regres])

    # apply mask to regression variables
    recons_regres = Lambda(lambda tensor: apply_mask(tensor, mask_regres),
                            mask=None,
                            name='lambda_regres',
                            output_shape=[sum(mask_regres)])(
                                    recons)

    # apply mask to binary variables
    recons_binary = Lambda(lambda tensor: apply_mask(tensor, mask_binary),
                            mask=None,
                            name='lambda_binary',
                            output_shape=[sum(mask_binary)])(
                                    recons) 

    # apply activations, linear for regressions, sigmoid for binaries
    recons_regres_activated = Activation('linear')(recons_regres) # this linear activation layer is a placeholder, doesn't do much!
    recons_binary_activated = Activation('sigmoid')(recons_binary)

    # instantiate decoder model
    encoder = Model(inputs  = inputs_encoder, 
                    outputs = [recons_regres_activated, recons_binary_activated],
                    name    = 'encoder')

    # Compile encoder
    encoder.compile(optimizer    = 'adadelta', 
                    loss         = ['mse', 'binary_crossentropy'],
                    loss_weights = [1.0, 1.0] )

    return encoder

#%%
################## FUNCTION TO SAVE KERAS MODELS ###########################

from contextlib import redirect_stdout

def save_model(model_object, filename, history, save_history=False, proj_root=proj_root):
    
    model_folders = ['SaveModels_AutoencCritic']

    # save keras model (ie serialize and send to file) 
    model_object.save(os.path.join(proj_root,*model_folders,filename+'_model.h5'))

    # save weights only (as backup)
    model_object.save_weights(os.path.join(proj_root,*model_folders,filename+'_weights.h5'))

    # save summary text
    filename_txt = os.path.join(proj_root,*model_folders,filename+'_summary.txt')
    with open(filename_txt, 'w') as f:
        with redirect_stdout(f):
            model_object.summary()
    
    # save graph image
    filename_png = os.path.join(proj_root,*model_folders,filename+'_graph.png')
    plot_model(model_object, to_file=filename_png, show_shapes=True)
   
    # save training history
    #if save_history:
    filename_history = os.path.join(proj_root,*model_folders,filename+'_history.npy')

    l_loss  = np.array(history.history['loss'])
    l_loss  = np.reshape(l_loss, (l_loss.shape[0],1))
    l_vloss = np.array(history.history['val_loss'])
    l_vloss = np.reshape(l_vloss, (l_vloss.shape[0],1))
        
    np.save(file = filename_history, arr = np.concatenate((l_loss, l_vloss), axis=1))

#%%
################## FUNCTION TO GRID SEARCH AUTOENCODER MODELS ###########################
from keras.callbacks import EarlyStopping

def grid_search(x_train, x_valid, params, train_type='std', y_train_mx=None, y_valid_mx=None):

    # extract useful params which are actually one value variables
    data_name          = params.get('data_name')[0]
    has_nan            = params.get('has_nan')[0]
    encode_layer_nodes = params.get('encode_layer_nodes') # is list, not single object

    earlystop_patience = params.get('earlystop_patience')[0]
    epochs             = params.get('epochs')[0]
    compile_function   = params.get('compile_function')

    #prep results table
    results = pd.DataFrame(columns=['batch_size', 'lky_alpha', 'batch_norm',
                                    'best_valid', 'matching_train', 'best_valid_epc'])

    # early stopping callback
    callback_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=earlystop_patience)

    loop_count = -1

    for leaky_alpha in params.get('leaky_alpha'):
        for apply_batchnorm in params.get('apply_batchnorm'):
            for batch_size in params.get('batch_size'):
        
                loop_count += 1
                                
                # filename for model instance
                filename = ('data_'   + str(data_name) + 
                            '_alpha_' + str(leaky_alpha) +
                            '_norm_'  + str(apply_batchnorm) +
                            '_batch_' + str(batch_size))

                # create model
                enc = compile_function(
                                    input_df           = x_train, 
                                    encode_layer_nodes = encode_layer_nodes, 
                                    has_nan            = has_nan,
                                    apply_batchnorm    = apply_batchnorm, 
                                    leaky_alpha        = leaky_alpha,
                                    kinit              = 'glorot_normal')
                
                if train_type == 'std':
                    #autoencoder outputs = input
                    y_train = [x_train.iloc[:,0:12], x_train.iloc[:,12: ]]
                    y_valid = [x_valid.iloc[:,0:12], x_valid.iloc[:,12: ]]
                else:
                    # not autoencoder, use y given as y_...mx
                    y_train = [y_train_mx.iloc[:,0:12], y_train_mx.iloc[:,12: ]]
                    y_valid = [y_valid_mx.iloc[:,0:12], y_valid_mx.iloc[:,12: ]]

                # fit model
                history = enc.fit(  
                                    x          = x_train,
                                    y          = y_train, # y=x for autoencoder
                                    epochs     = epochs, 
                                    batch_size = batch_size,
                                    verbose    = 2,
                                    shuffle    = True,
                                    validation_data = (x_valid, y_valid),
                                    callbacks  = [callback_es]
                                )

                # save models
                model_object, model_name = (enc, filename+'_enc')

                # save models, weights, graphs, training history, etc.
                save_model(model_object, model_name, history, save_history=True)

                # get result, i.e. min validation error and train error for same epoch
                best_valid      = min(history.history['val_loss'])
                best_valid_epc  = np.where(history.history['val_loss'] == best_valid)
                best_valid_epc  = best_valid_epc[0].item() #tie breaker, take first. need item else rtn array
                matching_train  = history.history['loss'][best_valid_epc]

                # save results to table for comparison
                results.loc[len(results)] = [batch_size, leaky_alpha, apply_batchnorm,
                                                best_valid, matching_train, best_valid_epc]

                print("loop: ", loop_count)
                print('\n')
                print(results.loc[len(results)-1])
                print('\n')
            
    # save to file because this took a long time to complete!
    results.to_csv(os.path.join(proj_root,'SaveModels_AutoencCritic','GridScan_Results_'+data_name+'.csv'))




#%%
################## GRID SEARCH HYPER PARAMS FOR MODELS ###########################

#####################
## For 10cols data ##
params =  { 
           # these params will be looped around in the grid, ie we'll try all unique combinations
           'leaky_alpha'       : [0.0, 0.4],
           'apply_batchnorm'   : [False, True],
           'batch_size'        : [32],
           'compile_function'  : define_and_compile,
           
           # these params will NOT be looped around, mostly lists of ONE object
           'data_name'         : ['10col'],
           'has_nan'           : [False],
           'encode_layer_nodes': [16, 8, 4],
           'earlystop_patience': [ 30],    # epochs to run before giving up on seeing better validation performance 
           'epochs'            : [200]
           } 
           
# do grid search, NB no return value. Function saves all results to file
grid_search(x_train = x_train_10cols,
            x_valid = x_valid_10cols,
            params  = params)


#%%
############ FUNCTION TO PLOT ALL TRAINING HISTORIES OR ALL EMBEDDINGS ##################################

import matplotlib.pyplot as plt
from keras.models import load_model

## Function to get list of folders in parent folder
def get_list_of_folders(parent_folder_name):
    mypath = os.path.join(parent_folder_name)
    folders_list = [f for f in listdir(mypath) if isdir(os.path.join(mypath, f))]
    folders_list_df = pd.DataFrame(folders_list, columns=['foldername'])
    folders_list_df['progress'] = 'Pending'
    del folders_list
    return folders_list_df

## Function to get list of all files in a given folder
def get_list_files_in_folder(folder_name):
    mypath = os.path.join(proj_root, folder_name)
    files_list = [f for f in listdir(mypath) if isfile(os.path.join(mypath, f))]
    files_list_df = pd.DataFrame(files_list, columns=['filename'])
    files_list_df['progress'] = 'Pending'
    del files_list
    #print(len(files_list_df.index))
    return files_list_df

## Function to plot all histories found in a folder of saved histories
def matrix_of_plots(data_name_5char='10col',       # '10col' or 'catego'
                    test_data_df=None,     # test data in pandas
                    plot_type='histories', # 'embeddings' or 'histories'
                    subfolder = ['SaveModels_AutoencCritic'],
                    charts_per_col = 2 # must be at least 2, else code errors
                    ):

    # get all files
    save_models    = get_list_files_in_folder(os.path.join(proj_root, *subfolder ))['filename']

    # filter down to training histories or embeddings
    if plot_type == 'histories':
        save_models = [file for file in save_models if file.endswith('_history.npy') and file.startswith('data_'+ data_name_5char)]
        prefix = 'Training histories for '
        xlabel = 'epochs'
        ylabel = 'loss'
    else: # embeddings
        save_models = [file for file in save_models if file.endswith('_model.h5') and file.startswith('data_' + data_name_5char)]
        prefix = 'Embeddings for '
        xlabel = 'latent x'
        ylabel = 'latent y' 

    # how many rows of training figures?
    rows = len(save_models) // charts_per_col
    cols = len(save_models) // rows

    # set size of matplotlib canvas
    plt.figure(num=None, figsize=(10, 10)) #dpi=80, facecolor='w', edgecolor='k'

    # set up the canvas
    canvas, figures = plt.subplots(rows, cols)

    #if only one row, then need to force row and column ref
    if len(figures.shape) == 1:
        figures = figures.reshape(1, figures.shape[0])

    # initialise row and column indices
    row = 0
    col = 0

    # loop around models, plotting each onto a subplot on the canvas
    for file_name in save_models:

        # make title
        lossfn_locn = file_name.find('lossfn')
        alpha_locn  = file_name.find('alpha')
        btchnm_locn = file_name.find('norm')

        loss_fun    = file_name[lossfn_locn+7:alpha_locn-1]
        alpha       = file_name[alpha_locn+6:alpha_locn+9]
        batchnorm   = file_name[btchnm_locn+5:btchnm_locn+8]

        title       = 'Ls:'+loss_fun+', Al:'+alpha+', Bn:'+batchnorm

        # setup subplot and title
        figures[row,col].set_title(title, size=7)
        figures[row,col].tick_params(labelsize=6)

        # space is limited, so only add axis labels at far left and foot of canvas, but not onto every subplot
        if col == 0:
            figures[row,col].set_ylabel(ylabel, size=5)
        if row == rows-1:
            figures[row,col].set_xlabel(xlabel, size=5)
        
        if plot_type == 'embeddings':

            # load model
            encoder = load_model(os.path.join(proj_root,*subfolder,file_name), compile=False)

            # get encodings (ie latent space, not reconstructions) for test data
            encoder_output = encoder.predict(test_data_df)

            # outputs from encoder = [z_mean, z_log_var, z]
            # so to see examples read the FIRST item in the list; z_mean
            # we don't want z (third item) because it has a random perturbation, we just want the mean
            z_vectors = encoder_output[0]

            # convert test embeddings to pandas
            z_vectors = pd.DataFrame(data=z_vectors, columns=['x','y'])

            # let's see if we can get a colour from the Turnover...
            z_vectors['turnover_curr'] = test_data_df['001_SOI_TurnoverRevenue_curr'].reset_index(drop=True)

            # plot it
            figures[row,col].scatter(x=z_vectors['x'], y=z_vectors['y'], c=z_vectors['turnover_curr'])

        else: # training history

            # load data
            tng_history = np.load(os.path.join(proj_root,*subfolder,file_name))

            # plot it
            figures[row,col].plot(tng_history)

        # get next row and column
        col += 1
        if col > cols-1:
            col = 0
            row += 1

    # try to prevent overlap between subplot titles, axis labels and ticks
    canvas.tight_layout()
    # add title
    canvas.suptitle(prefix + data_name_5char + ' data.', fontsize=11)
    # ensure canvas title does not overlap plots
    canvas.subplots_adjust(top=0.88) 

    # save result
    plt.savefig(os.path.join(proj_root, *subfolder, prefix + data_name_5char + ' data.jpg'),  dpi=300)

#%%
############ INSPECT TRAINING ON 10COLS DATA #####################

matrix_of_plots(data_name_5char='10col', plot_type='histories')

# These models all trained reasonably well
# Binary cross entropy experienced the greatest drop in loss
#   Alpha 0.2, BatchNorm = False
# Complex loss fn also trained well, the best hyperparams were:
#   Alpha 0.0, BatchNorm = False

#%%
# Get data for combined model

file_name = 'data_10col_lossfn_mixed_alpha_0.2_norm_False_batch_32_autoenc_model.h5'
autoencoder = load_model(filepath = os.path.join(proj_root,'SaveModels_Autoenc','TwoLossModel',file_name), 
                         compile  = False)

file_name = 'data_10col_lossfn_mixed_alpha_0.2_norm_False_batch_32_enc_model.h5'
autoencoder_encoder = load_model(filepath = os.path.join(proj_root,'SaveModels_Autoenc','TwoLossModel',file_name), 
                                 compile  = False)

# custom lambda layers cause problem loading keras model from file
# so define model and load weights
critic_encoder = define_and_compile(
                        input_df           = x_train_10cols, 
                        encode_layer_nodes = [16,8,4], 
                        has_nan            = False,
                        apply_batchnorm    = False, 
                        leaky_alpha        = 0.4,
                        kinit              = 'glorot_normal')

file_name = 'data_10col_alpha_0.4_norm_False_batch_32_enc_weights.h5'
critic_encoder.load_weights(os.path.join(proj_root,'SaveModels_AutoencCritic','16_8_4',file_name))

# training data
x_train_latent = autoencoder_encoder.predict(x_train_10cols)
x_train_recons = np.concatenate(autoencoder.predict(x_train_10cols), axis=1)
x_train_critic = np.concatenate(critic_encoder.predict(x_train_recons), axis=1)

x_train_mixed = np.concatenate([x_train_latent, x_train_recons, x_train_critic], axis=1)

# validation data
x_valid_latent = autoencoder_encoder.predict(x_valid_10cols)
x_valid_recons = np.concatenate(autoencoder.predict(x_valid_10cols), axis=1)
x_valid_critic = np.concatenate(critic_encoder.predict(x_valid_recons), axis=1)

x_valid_mixed = np.concatenate([x_valid_latent, x_valid_recons, x_valid_critic], axis=1)

#%%
# save graph image
filename_png = os.path.join(proj_root,'SaveModels_AutoencCritic',file_name+'_graphtest.png')
plot_model(critic_encoder, to_file=filename_png, show_shapes=True)

#%%
# COMBINED MODEL
############################################

# Inputs: 
#   Autoencoder's Latent space vector
#   Autoencoder's reconstruction
#   Critic's reconstruction of reconstruction
# Outputs
#   New reconstruction
# Targets
#   Autoencoder's original input data, ie x_train

# Can we reduce reconstruction error by using the critic?

def define_and_compile_mx( input_df,
                        encode_layer_nodes,
                        has_nan             = False,
                        apply_batchnorm     = False,
                        leaky_alpha         = 0.2,
                        kinit               = 'glorot_normal'):

    ## Instantiate inputs ##
    original_dim = len(input_df.columns)

    # Input to Encoder (from data to be modelled)
    inputs_encoder = Input(shape=(original_dim,), name='encoder_input')

    encoded = inputs_encoder
    loop_count = 0  

    # for each Lambda layer there is an encoder:
    for layer_nodes in encode_layer_nodes:
        
        # define layer
        encoded = Dense(
                        units = layer_nodes, 
                        name = 'dense_lr_'+str(loop_count)+'_'+str(layer_nodes),
                        kernel_initializer=kinit)(
                            encoded)

        # optionally apply batch norm
        if(apply_batchnorm):
            encoded = BatchNormalization()(encoded)
        
        # define activation
        encoded = LeakyReLU(alpha= leaky_alpha, 
                            name = 'lkyrelu_lyr_'+str(loop_count)+'_'+str(layer_nodes))(
                                encoded)

        loop_count += 1
        # END OF FOR LOOP

    # the final layer
    # Must have same number sof outputs as there are inputs
    # No activation is applied at this stage, that comes after another lambda layer
    # The value eventually gets linear activiation. The binary gets sigmoid
    encoded = Dense(units = 22, 
                    name = 'dense_final',
                    kernel_initializer=kinit)(
                            encoded)

    # END OF FOR LOOP

    # apply mask to regression variables
    recons_regres = Lambda(lambda x: x[:,0:12], 
                            name='lambda_regres',
                            output_shape=[12])(
                                    encoded)

    # apply mask to binary variables
    recons_binary = Lambda(lambda x: x[:,12:],
                            name='lambda_binary',
                            output_shape=[10])(
                                    encoded) 

    # apply activations, linear for regressions, sigmoid for binaries
    recons_regres_activated = Activation('linear')(recons_regres) # this linear activation layer is a placeholder, doesn't do much!
    recons_binary_activated = Activation('sigmoid')(recons_binary)

    # instantiate decoder model
    encoder = Model(inputs  = inputs_encoder, 
                    outputs = [recons_regres_activated, recons_binary_activated],
                    name    = 'encoder')

    # Compile encoder
    encoder.compile(optimizer    = 'adadelta', 
                    loss         = ['mse', 'binary_crossentropy'],
                    loss_weights = [1.0, 1.0] )

    return encoder

#%%
# EXECUTE NEW MIXED MODEL
######################
## For 10cols data ##

mixed_model = define_and_compile_mx(
                        input_df           = pd.DataFrame(x_train_mixed), 
                        encode_layer_nodes = [64,64,32], 
                        has_nan            = False,
                        apply_batchnorm    = False, 
                        leaky_alpha        = 0.0,
                        kinit              = 'glorot_normal')

history = mixed_model.fit(  
                    x          = x_train_mixed,
                    y          = [x_train_10cols.iloc[:,0:12], x_train_10cols.iloc[:,12: ]],
                    epochs     = 200, 
                    batch_size = 32,
                    verbose    = 2,
                    shuffle    = True,
                    validation_data = (x_valid_mixed, [x_valid_10cols.iloc[:,0:12], x_valid_10cols.iloc[:,12: ]])
                                )

save_model(mixed_model, 'mixed_', history, save_history=True, proj_root=proj_root)

# plot history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%
############ RECONSTRUCTION ERRORS #####################

# Let's compare reconstruction errors for test data vs for random figures
# We'd expect the autoencoder to reconstruct the test data with a much smaller error
# This would make it useful in distinguishing random data from real data

#%%
############ FUNCTIONS TO MAKE RANDOM REPORTS #####################
from scipy.stats import truncnorm

def get_truncated_normal(qty, mean=0, sd=1, minimum=-3, maximum=3):

    #get truncated normal (ie normal distrib within a set max and min)
    distribution = truncnorm(a     = (minimum - mean) / sd, 
                             b     = (maximum - mean) / sd, 
                             loc   = mean, 
                             scale = sd
                             )

    # get samples from that distribution
    samples = distribution.rvs(qty)

    return samples

def generate_fake_samples_fromRandoms(dataset, qty):

    # get max and min for each column
    maximums = dataset.max()
    minimums = dataset.min()
    means    = dataset.mean()
    sds      = dataset.std()

    # generate gauusian random data (noise) for each column
    # BUT do so WITHIN the range of the actual data (min, max)
    fake_samples_x = [get_truncated_normal(qty, mean=mean, sd=sd, minimum=mini, maximum=maxi) 
                      for mini, maxi, mean, sd 
                      in zip(minimums, maximums, means, sds)]

    fake_samples_x = np.stack(fake_samples_x, axis=1)

    return fake_samples_x

#%%
############ FUNCTIONS TO CALCULATE RECONSTRUCTION ERRORS #####################
from sklearn.metrics import log_loss

def get_reconstruction_losses(inputs, outputs, has_nan=False, recon_loss_method='complex'):
    # basically the same approach as the reconstruction loss in the custom loss function (see above)
    # except using numpy, as opposed to Keras backend
    # see the above custom loss function for notes
    original_dim = inputs.shape[1]

    if not has_nan :
        colqty = (original_dim - 2)/2
    else :
        colqty = (original_dim - 2)/3

    colqty = int(colqty)

    # error in value
    reconstruction_abs_errors_A = np.abs(inputs[:,0:2] - outputs[:,0:2])

    inputs_regression   = inputs[:,2:2+colqty]
    outputs_regression  =outputs[:,2:2+colqty]
    reconstruction_abs_errors_B = np.abs(inputs_regression - outputs_regression)

    # double error in value IF there is error in category
    inputs_categ_ispos  = inputs[:,2+colqty:2+2*colqty]
    outputs_categ_ispos =outputs[:,2+colqty:2+2*colqty]

    ispos_abs_errors = np.abs(outputs_categ_ispos - inputs_categ_ispos)
    ispos_abs_errors = ispos_abs_errors + 1

    reconstruction_abs_errors_B = reconstruction_abs_errors_B * ispos_abs_errors
    concat = np.concatenate((reconstruction_abs_errors_A , reconstruction_abs_errors_B), axis=1)

    reconstruction_losses = np.mean(concat, axis=1)

    return reconstruction_losses


#%%
# Get Reconstruction Errors for real data.
# We get the mean error from 500 samples of real data and 500 sample sof fake data
# then compare the distributions of the two samples
# This will then allow us to score a batch of reports as being either fake or real

for use_mixed_model in [True, False]:

    recon_errs_real_mean = []
    recon_errs_fake_mean = []

    for i in range(1,500):
        # Get Reconstruction Errors for real data
        # assume 150 examples summarized, as would be used in GAN training
        # use training set, not test, because GAN will be using training set
        inputs  = np.array(x_testi_10cols.sample(32))
        # get outputs, there are two (regression and binary) so concatenate them using hstack
        if use_mixed_model:
            x_recon_latent = autoencoder_encoder.predict(inputs)
            x_recon_recons = np.hstack(autoencoder.predict(inputs))
            x_recon_critic = np.hstack(critic_encoder.predict(x_recon_recons))
            x_recon_mixed  = np.concatenate([x_recon_latent, x_recon_recons, x_recon_critic], axis=1)

            outputs = np.hstack(mixed_model.predict(x_recon_mixed))
        else:
            outputs = np.hstack(autoencoder.predict(inputs))

        recon_errs_real = get_reconstruction_losses(inputs  = inputs,outputs = outputs)
        
        # Get Reconstruction Errors for fake data
        inputs  = generate_fake_samples_fromRandoms(dataset = x_train_10cols,qty = 32)
        # get outputs, there are two (regression and binary) so concatenate them using hstack
        if use_mixed_model:
            x_recon_latent = autoencoder_encoder.predict(inputs)
            x_recon_recons = np.hstack(autoencoder.predict(inputs))
            x_recon_critic = np.hstack(critic_encoder.predict(x_recon_recons))
            x_recon_mixed  = np.concatenate([x_recon_latent, x_recon_recons, x_recon_critic], axis=1)

            outputs = np.hstack(mixed_model.predict(x_recon_mixed))
        else:
            outputs = np.hstack(autoencoder.predict(inputs))

        recon_errs_fake = get_reconstruction_losses(inputs  = inputs,outputs = outputs)

        recon_errs_real_mean.append(np.mean(recon_errs_real))
        recon_errs_fake_mean.append(np.mean(recon_errs_fake))

    ## Plot mean reconstruction errors as overlaid histograms

    bins = np.linspace(0, 1.5, 50)
    plt.hist(recon_errs_real_mean, bins, alpha=0.5, label='real')
    plt.hist(recon_errs_fake_mean, bins, alpha=0.5, label='fake')
    plt.legend(loc='upper right')
    plt.title('Mean Error in 500 Fake vs Real Reconstructions')
    plt.show()

    print("Mean error in batch of real data: ", np.mean(np.array(recon_errs_real_mean)) )
    print("Mean error in batch of fake data: ", np.mean(np.array(recon_errs_fake_mean)) )


#%%
# RESULT
# The lower the error the better, happily the fake and real errors distirbutions don't overlap!
# If the mean error in a batch is around 0.35 then data is indistinguishable from real
# If the mean error in a batch is around 1.0 then data is indistinguishable from fake
# Any batch with a mean error between 0.6 and 1.0 is a mix of fake and real reports
# This is a useful scoring mechanism for our VAE and GAN






#%%
