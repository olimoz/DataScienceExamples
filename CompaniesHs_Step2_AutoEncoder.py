

#%%
import os
import pandas as pd
import numpy as np
import re
from os import listdir, chdir
from os.path import isfile, isdir, join
from dfply import *
import time
import datetime
from bs4 import BeautifulSoup, SoupStrainer
from ggplot import *
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization, Concatenate
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras import metrics
from keras import backend as K
from keras.utils import plot_model

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
# CUSTOM LOSS FUNCTION

# Note how this custom loss function returns a function, 
# which is where the nested function uses its access to the arguments of the enclosing function...aka 'closure'
# Defining it this way allows us to provide the loss function with more than the standard
# loss function variables of inputs and outputs
# see https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618

def model_loss(original_dim, has_nan, recon_loss_method):

    def autoenc_loss(inputs, outputs):

        # this is a combination of regression fields (use mse?) and binary category field (use binary cross entropy?)
        # if the data is not sparse (does not have nan's), then we have 2+n regression fields + n categoricals which denote whether the field is negative or not.
        # The first two fields are Duration and Date. They have no categoricals associated with them
        
        # There are three ways the reconstruction loss could be calculated. These ar eoptions at compile time
        # for two of the methods we need to know which columns are regressional and which categorical

        if recon_loss_method == 'bxent':
            # the next simplest reconstruction loss is binary cross entropy (typically used in categoricals)
            # without ANY acknowledgement that some fields are regressional and others categorical
            reconstruction_loss = metrics.binary_crossentropy(inputs, outputs)

        elif recon_loss_method == 'complex':
            # the most complex method attempts to increase regression error where there is a category error. 
            # Eg if regression field is accurate, but of the wrong sign (+ve, rather than -ve)
            # then the regression error is actually twice as big.

            # This depends on whether there are fields suffixed 'isnan'
            if not has_nan :
                # First we split out input and output tensors into the regression and categorical fields...
                colqty = (original_dim - 2)/2
            else :
                colqty = (original_dim - 2)/3

            # tensor indices must be integers
            colqty = int(colqty)

            # lets get the regression loss on those first two fields
            reconstruction_abs_errors_A = K.abs(K.slice(inputs, start=[0,0], size=[-1,2]) 
                                                - 
                                                K.slice(outputs,start=[0,0], size=[-1,2]))

            # get the regression fields...
            inputs_regression  = K.slice(inputs, start=[0,2], size=[-1,colqty])
            outputs_regression = K.slice(outputs,start=[0,2], size=[-1,colqty])

            # get the ispos categorical fields
            inputs_categ_ispos  = K.slice(inputs, start=[0,colqty+2], size=[-1,colqty])
            outputs_categ_ispos = K.slice(outputs,start=[0,colqty+2], size=[-1,colqty])

            # let's get our first estimate of the regression error
            reconstruction_abs_errors_B = K.abs(inputs_regression - outputs_regression)

            # BUT the categoricals indicate whether a regression field is +ve or -ve
            # So, we double the regression error on a field if categorical (ispos) is incorrect for that field.
            ispos_abs_errors = K.abs(outputs_categ_ispos - inputs_categ_ispos)
            # ispos_abs_errors = 0 if no error and 1 if error.
            # but we need 1 if no error and 2 if error
            ispos_abs_errors = ispos_abs_errors + 1

            # so we simply multiply the outputs by ispos_abs_errors
            reconstruction_abs_errors_B = reconstruction_abs_errors_B * ispos_abs_errors
            
            if has_nan :
                # its somewhat different if there is sparse data. We now have fields which indicate whether we have data for a field or not
                # if we don't have data then we should ignore the error, as we don't know what the value should be
                # so, in effect we use those 'isnan' fields to set the calculated error to zero
                # get the regression fields...

                # get the isnan categorical fields
                inputs_categ_isnan  = K.slice(inputs, start=[0,2*colqty+2], size=[-1,colqty])
                outputs_categ_isnan = K.slice(outputs,start=[0,2*colqty+2], size=[-1,colqty])

                # get the error in isnan
                isnan_abs_errors = K.abs(inputs_categ_isnan - outputs_categ_isnan)

                # ispos_abs_errors = 0 if no error and 1 if error.
                # but we need 1 if no error and 2 if error
                isnan_abs_errors = isnan_abs_errors + 1

                # where isnan_errors = 1 then we add to the regression error
                # ie if we report a figure where non was required, or vice versa, then we add that amount to the error
                reconstruction_abs_errors_B = reconstruction_abs_errors_B * isnan_abs_errors

            # stack errors before calculating mean
            concat = K.concatenate((reconstruction_abs_errors_A , reconstruction_abs_errors_B), axis=1)

            # then we convert the matrix of errors into a single average loss figure for each example (row)
            reconstruction_loss = K.mean(concat, axis=1)

        else :
            print("ERROR IN RECONSTRUCTION LOSS SELECTION")
            # intentionally break it! Yeehah, cowboy coding....
            return None
        
        return reconstruction_loss

    # For loss calculation keras expects a function to be returned
    return autoenc_loss

#%%

def define_and_compile_ac(  input_df,
                            encode_layer_nodes,
                            decode_layer_nodes,
                            reconstruction_cols,
                            recon_loss_method,
                            loss_function       = model_loss,
                            has_nan             = False,
                            latent_space_nodes  = 2,
                            apply_batchnorm     = False,
                            leaky_alpha         = 0.2,
                            kinit               = 'glorot_normal'):

    ####################################################################
    # Encoder
    ####################################################################   
    ## Instantiate inputs ##
    # we'll start with the simplest model, 10cols
    original_dim = len(input_df.columns)

    # Input to Encoder (from data to be modelled)
    inputs_encoder = Input(shape=(original_dim,), name='encoder_input')

    # Encode Hidden Layers
    encoded = inputs_encoder

    for nodes in encode_layer_nodes:
        
        # define layer
        encoded = Dense(nodes, kernel_initializer=kinit, name='encode_nodes_'+str(nodes))(encoded)
        
        # optionally apply batch norm
        if(apply_batchnorm):
            encoded = BatchNormalization()(encoded)
        
        # define activation
        encoded = LeakyReLU(alpha=leaky_alpha, name='lkyrelu_encode_nodes_'+str(nodes))(encoded)
    
    ####################################################################
    # Latent Space
    ####################################################################

    latent_space_embeddings = Dense(latent_space_nodes, kernel_initializer=kinit, name='latent_space')(encoded)
    
    # having defined layers, create graph
    #   inputs = input batch
    #   output = latent space embedding for each record in batch
    encoder = Model(inputs=inputs_encoder, outputs=latent_space_embeddings, name='encoder')

    ####################################################################
    # Decoder
    ####################################################################

    # Input to Decoder (from latent space)
    inputs_decoder = Input(shape=(latent_space_nodes,), name='decoder_input')

    # Decoder Hidden Layers 
    decoded = inputs_decoder

    for nodes in decode_layer_nodes:
    
        # define layer
        decoded = Dense(nodes, kernel_initializer=kinit, name='decode_nodes_' + str(nodes))(decoded)
        
        # optionally apply batch norm
        if(apply_batchnorm):
            decoded = BatchNormalization()(decoded)
        
        # define activation
        decoded = LeakyReLU(alpha=leaky_alpha, name='lkyrelu_decode_nodes_' + str(nodes))(decoded)

    # Reconstruction
    ####################################################################
    # We must now reconstruct the input, which is a combination (ie concatenation)
    # of values and binaries (..._notna and ..._ispos)
    # define empty list of layers
    reconstruction_layers = []

    # we now append layers into this list by iterating over the columns of the input data
    # we want one layer for each column in the input data, so output matches input
    # columns with binary data are handled differently to columns with value data
    for index, row in reconstruction_cols.iterrows():

        nm = row['col_name']

        if row['col_type'] == 'value':
            # define layer to reconstitute a value, intentionally no activation
            reconstruction_layers.append( Dense(1, kernel_initializer=kinit, name=nm)(decoded) )
        
        else : # it must be binary
            # define layer to reconstitute a binary, uses sigmoid activation
            reconstruction_layers.append( Dense(1, kernel_initializer=kinit, activation='sigmoid', name=nm)(decoded) )

    # concatenate into reconstruction
    reconstruction = concatenate(reconstruction_layers)

    # instantiate decoder model
    decoder = Model(inputs = inputs_decoder, output = reconstruction, name = 'decoder')

    ####################################################################
    # Autoencoder = Encoder + Decoder
    ####################################################################

    # DECODE FROM LATEBT SPACE & RECONSTRUCT
    outputs_decoder = decoder(latent_space_embeddings)

    # Autoencoder = Encoder + Decoder
    autoencoder = Model(inputs = inputs_encoder, outputs = outputs_decoder, name = 'autoencoder')

    # Compile autoencoder
    autoencoder.compile(optimizer = 'adadelta', loss = loss_function(original_dim, has_nan, recon_loss_method))

    return encoder, decoder, autoencoder

#%%
################## FUNCTION TO SAVE KERAS MODELS ###########################

from contextlib import redirect_stdout

def save_model(model_object, filename, history, save_history=False, proj_root=proj_root):
    
    # save keras model (ie serialize and send to file) 
    model_object.save(os.path.join(proj_root,'SaveModels_Autoenc',filename+'_model.h5'))

    # save weights only (as backup)
    model_object.save_weights(os.path.join(proj_root,'SaveModels_Autoenc',filename+'_weights.h5'))

    # save summary text
    filename_txt = os.path.join(proj_root,'SaveModels_Autoenc',filename+'_summary.txt')
    with open(filename_txt, 'w') as f:
        with redirect_stdout(f):
            model_object.summary()
    
    # save graph image
    filename_png = os.path.join(proj_root,'SaveModels_Autoenc',filename+'_graph.png')
    plot_model(model_object, to_file=os.path.join(proj_root,'SaveModels',filename_png), show_shapes=True)
   
    # save training history
    #if save_history:
    filename_history = os.path.join(proj_root,'SaveModels_Autoenc',filename+'_history.npy')

    l_loss  = np.array(history.history['loss'])
    l_loss  = np.reshape(l_loss, (l_loss.shape[0],1))
    l_vloss = np.array(history.history['val_loss'])
    l_vloss = np.reshape(l_vloss, (l_vloss.shape[0],1))
        
    np.save(file = filename_history, arr = np.concatenate((l_loss, l_vloss), axis=1))

#%%
################## FUNCTION TO IDENTIFY COLUMN TYPE ###########################
# The decoder demands we know which columns are values and which are binary categories
# We'll use column names to deduce this, but without special characters which are not permitted in keras

def get_col_spec(df):
    '''
        TAKES  : the input dataframe
        RETURNS: a dataframe listing column names (without special characters, useful in Keras) and its type
    '''
    colspec = ['binary' if colname.endswith('_ispos') or colname.endswith('_notna') else 'value' for colname in df.columns]
    colname = [re.sub(r'[^\w]', '', colname) for colname in df.columns]
    colspec_df = pd.DataFrame({'col_name' : colname, 'col_type': colspec})
    return colspec_df


#%%
################## FUNCTION TO GRID SEARCH VAE MODELS ###########################
from keras.callbacks import EarlyStopping

def grid_search(x_train, x_valid, params):

    # extract useful params which are actually one value variables
    data_name          = params.get('data_name')[0]
    has_nan            = params.get('has_nan')[0]
    encode_layer_nodes = params.get('encode_layer_nodes') # is list, not single object
    decode_layer_nodes = params.get('decode_layer_nodes') # is list, not single object
    latent_space_nodes = params.get('latent_space_nodes')[0]
    earlystop_patience = params.get('earlystop_patience')[0]
    epochs             = params.get('epochs')[0]

    #prep results table
    results = pd.DataFrame(columns=['recon_loss_method', 'batch_size', 'lky_alpha', 'batch_norm',
                                    'best_valid', 'matching_train', 'best_valid_epc'])

    # early stopping callback
    callback_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=earlystop_patience)

    loop_count = -1

    for recon_loss_method in params.get('recon_loss_method'):
        for leaky_alpha in params.get('leaky_alpha'):
            for apply_batchnorm in params.get('apply_batchnorm'):
                for batch_size in params.get('batch_size'):
            
                    loop_count += 1
                                    
                    # filename for model instance
                    filename = ('data_'   + str(data_name) + 
                                '_lossfn_'+ str(recon_loss_method) + 
                                '_alpha_' + str(leaky_alpha) +
                                '_norm_'  + str(apply_batchnorm) +
                                '_batch_' + str(batch_size))

                    # create model
                    enc, dec, autoenc = define_and_compile_ac(
                                        input_df           = x_train, 
                                        encode_layer_nodes = encode_layer_nodes, 
                                        decode_layer_nodes = decode_layer_nodes,
                                        reconstruction_cols= get_col_spec(x_train),
                                        latent_space_nodes = latent_space_nodes, 
                                        recon_loss_method  = recon_loss_method,
                                        loss_function      = model_loss,
                                        has_nan            = has_nan,
                                        apply_batchnorm    = apply_batchnorm, 
                                        leaky_alpha        = leaky_alpha,
                                        kinit              = 'glorot_normal')
                    # fit model
                    history = autoenc.fit(  
                                        x          = x_train,
                                        y          = x_train, # y=x for autoencoder
                                        epochs     = epochs,
                                        batch_size = batch_size,
                                        verbose    = 2,
                                        shuffle    = True,
                                        validation_data = (x_valid, x_valid),
                                        callbacks  = [callback_es]
                                    )

                    # save models
                    models = [(enc, filename+'_enc'), (dec, filename+'_dec'), (autoenc, filename+'_autoenc')]

                    for each_model in models:
                        model_object, model_name = each_model

                        #only save history for autoenc, not enc and not dec
                        if model_name[-7:] == 'autoenc':
                            save_history = True
                        else :
                            save_history = False

                        # save models, weights, graphs, training history, etc.
                        save_model(model_object, model_name, history, save_history)

                    # get result, i.e. min validation error and train error for same epoch
                    best_valid      = min(history.history['val_loss'])
                    best_valid_epc  = np.where(history.history['val_loss'] == best_valid)
                    best_valid_epc  = best_valid_epc[0].item() #tie breaker, take first. need item else rtn array
                    matching_train  = history.history['loss'][best_valid_epc]

                    # save results to table for comparison
                    results.loc[len(results)] = [recon_loss_method,   batch_size,    leaky_alpha, apply_batchnorm,
                                                best_valid, matching_train, best_valid_epc]

                    print("loop: ", loop_count)
                    print('\n')
                    print(results.loc[len(results)-1])
                    print('\n')
            
    # save to file because this took a long time to complete!
    results.to_csv(os.path.join(proj_root,'SaveModels_Autoenc','GridScan_Results_'+data_name+'.csv'))

#%%
################## GRID SEARCH HYPER PARAMS FOR MODELS ###########################

#####################
## For 10cols data ##
params =  { 
           # these params will be looped around in the grid, ie we'll try all unique combinations
           'recon_loss_method' : ['bxent', 'complex'],
           'leaky_alpha'       : [0.4, 0.2, 0.0],
           'apply_batchnorm'   : [False, True],
           'batch_size'        : [32],
           
           # these params will NOT be looped around, mostly lists of ONE object
           'data_name'         : ['10col'],
           'has_nan'           : [False],
           'encode_layer_nodes': [16, 8,  4],
           'decode_layer_nodes': [4,  8, 16],
           'latent_space_nodes': [2],     # code cannot yet handle 3 or more latent dims. So choose 2!
           'earlystop_patience': [30],    # epochs to run before giving up on seeing better validation performance 
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
def matrix_of_plots(data_name_5char,       # '10col' or 'catego'
                    test_data_df=None,     # test data in pandas
                    plot_type='histories', # 'embeddings' or 'histories'
                    subfolder = 'SaveModels_Autoenc',
                    charts_per_col = 4
                    ):

    # get all files
    save_models    = get_list_files_in_folder(os.path.join(proj_root, subfolder ))['filename']

    # filter down to training histories or embeddings
    if plot_type == 'histories':
        save_models = [file for file in save_models if file[-19:]=='autoenc_history.npy' and file[0:10] == 'data_' + data_name_5char]
        prefix = 'Training histories for '
        xlabel = 'epochs'
        ylabel = 'loss'
    else: # embeddings
        save_models = [file for file in save_models if file[-16:]=='autoenc_model.h5' and file[0:10] == 'data_' + data_name_5char]
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
            encoder = load_model(os.path.join(proj_root,subfolder,file_name), compile=False)

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
            tng_history = np.load(os.path.join(proj_root,subfolder,file_name))

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
    plt.savefig(os.path.join(proj_root, subfolder, prefix + data_name_5char + ' data.jpg'),  dpi=300)

#%%
############ INSPECT TRAINING ON 10COLS DATA #####################

matrix_of_plots(data_name_5char='10col', plot_type='histories')

# These models all trained reasonably well
# Binary cross entropy experienced the greatest drop in loss
#   Alpha 0.2, BatchNorm = False
# Complex loss fn also trained well, the best hyperparams were:
#   Alpha 0.0, BatchNorm = False

file_name = 'data_10col_lossfn_bxent_alpha_0.2_norm_False_batch_32_autoenc_model.h5'
autoencoder = load_model(os.path.join(proj_root,'SaveModels_Autoenc',file_name), compile=False)

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

recon_errs_real_mean = []
recon_errs_fake_mean = []

for i in range(1,500):
    # Get Reconstruction Errors for real data
    # assume 150 examples summarized, as would be used in GAN training
    # use training set, not test, because GAN will be using training set
    inputs  = np.array(x_train_10cols.sample(150))
    outputs = autoencoder.predict(inputs)
    recon_errs_real = get_reconstruction_losses(inputs  = inputs,outputs = outputs)
    
    # Get Reconstruction Errors for fake data
    inputs  = generate_fake_samples_fromRandoms(dataset = x_train_10cols,qty = 150)
    outputs = autoencoder.predict(inputs)
    recon_errs_fake = get_reconstruction_losses(inputs  = inputs,outputs = outputs)

    recon_errs_real_mean.append(mean(recon_errs_real))
    recon_errs_fake_mean.append(mean(recon_errs_fake))

## Plot mean reconstruction errors as overlaid histograms

bins = np.linspace(0, 1.5, 50)
plt.hist(recon_errs_real_mean, bins, alpha=0.5, label='real')
plt.hist(recon_errs_fake_mean, bins, alpha=0.5, label='fake')
plt.legend(loc='upper right')
plt.title('Mean Error in 150 Fake vs Real Reconstructions')
plt.show()

#%%
# RESULT
# The lower the error the better, happily the fake and real errors distirbutions don't overlap!
# If the mean error in a batch is around 0.6 then data is indistinguishable from real
# If the mean error in a batch is around 1.0 then data is indistinguishable from fake
# Any batch with a mean error between 0.6 and 1.0 is a mix of fake and real reports
# This is a useful scoring mechanism for our VAE and GAN

print("Mean error in batch of real data: ", mean(np.array(recon_errs_real_mean)) )
print("Mean error in batch of fake data: ", mean(np.array(recon_errs_fake_mean)) )


#%%
#####################
## For catego data ##
params =  { 
           # these params will be looped around in the grid
           'recon_loss_method' : ['simple_mse', 'simple_bxent', 'medium', 'complex'],
           'leaky_alpha'       : [0.4, 0.2, 0.0],
           'apply_batchnorm'   : [False],
           'batch_size'        : [100],   # batch must be >=100 for vae, due use of monte carlo for loss (see link to tiao.io)
           'has_nan'           : [True],

           # these params will NOT be looped around, mostly lists of ONE object
           'data_name'         : ['catego'],
           'encode_layer_nodes': [256, 128, 64, 32, 16,   8,   4],
           'decode_layer_nodes': [  4,   8, 16, 32, 64, 128, 256],
           'latent_space_nodes': [2], # code cannot yet handle 3 or more latent dims. So choose 2!
           'earlystop_patience': [20],# epochs to run before giving up on seeing better validation performance 
           'epochs'            : [200]
           } 

grid_search(x_train = x_train_catego, 
            x_valid = x_valid_catego,
            params  = params)


