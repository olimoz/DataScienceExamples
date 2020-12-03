#%%
### VAE LATENT SPACE AS DATA FOR SEQ 2 SEQ MODEL

# So far we have considered a latent space of company reports.
# But many companies present us with a time series of company reports, 
# not just one-off reports.

# We could build a latent space of time series of company reports, 
# thereby one point in the latent space represents a trajectory, a time-series
# this is the trajectory of each company's results

# At first it is tempting to consider a latent space with a handful of dimensions 
# for the compressed features but also with a time dimension, undoctored (uncompressed)
# This would allows us to easily view the trajectories in latent space

# BUT

# This is effectively what we already have using a latent space of single reports.
# We simply need to select those points by company and draw the trajectory over time

# The proposal here is different. It is a latent space whany reports. It will necessarily be of a higher number of dimensions
# than the 2D latent space used in the single reports latent space

# To achieve this new latent space we need a model which can encode time series.
# LSTM's have been chosen although 1D convs are a possible alternative.

# LSTM autoencoders with meaningful latent spaces are the root of the architecture for a 
# number of powerful NLP tools. Those may also involve bi-directional LSTM's and 'attention'.
# Di-directional analysis isn't required for our problem.
# It would be attractive to use some form of attention to improve accuracy but
# attention feeds information from the encoder's activations to the decoder's layers
# Yet in a generative model we assume we don't have the encoder nor the sequence fed to it, 
# and hence don't have the encoder activations to feed into the decoder.

# The Decoder of our model is trained to repreduce the entire series. 
# It doesn't have to, it could be trained to give the first or last in the series
# or even a prediction for the next in the series.

# As we are dealing with generativ emodels we need to consder how to enforce structure
# in the latent space. Here we use 
#   the VAE (gaussian structure) and 
#   the MMD (overlapping gaussians structure)

# Data Limits
# The below analysis shows that we don't really have enough companies 
# presenting a continuous sequence of company reports such that we can train LSTMs
# So the code has been developed and debugge,d but performance not analysed
# Collecting more data, possibly from the SEC, is a more immediate priority.

#%%

### ALTERNATIVE APPROACH ###

# It is possible to return to plotting individual reports in latent space
# Then train another model to navigate that latent space from a to b
# If the latent space is say, 6 dimensional, then this may be a useful approach
# Using the sequences of reports for a single company we could train 
# a model to find 'time like' paths from a to b (first report to last)
# in the latent space.
# With sufficient data we could train a model to classify those paths in latent space
# as; growing, failing, static etc.
# Having done so, we can use a decoder to generate possible paths and their likelihoods


#%%
#### IMPORTS
import os
import cProfile
import re

import math       as math
import pandas     as pd
import numpy      as np
import tensorflow as tf

from tensorflow.keras        import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, LeakyReLU, BatchNormalization, Concatenate, Reshape, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils  import plot_model

#%% 
#### Confirm environment

print("Tensorflow Version:", tf.__version__)
print("Is tensorflow executing eagerly? ", tf.executing_eagerly())
print("Is tensorflow using GPU? ", tf.test.is_gpu_available())
print("Is tensorflow using Cuda? ", tf.test.is_built_with_cuda())

# This model will work in float 32
# but if you want float 64, uncomment this line:
# tf.keras.backend.set_floatx('float64')

#%%
### LOAD DATA

proj_root  = 'E:\\Data\\SEC'
subfolders = ['SaveModels_VAE_TS_SEC']
filename   = 'VAE_TS_SEC'
cols_regres= 11
cols_binary=  9

prepped_10cols = pd.read_csv(os.path.join(proj_root, 'prepped_10cols.csv'))
sds_10cols     = pd.read_csv(os.path.join(proj_root, 'sds_10cols.csv'), index_col=0)
means_10cols   = pd.read_csv(os.path.join(proj_root, 'means_10cols.csv'), index_col=0)

x_train_10cols = pd.read_csv(os.path.join(proj_root, 'x_train_10cols.csv'), index_col='Report_Ref')
x_valid_10cols = pd.read_csv(os.path.join(proj_root, 'x_valid_10cols.csv'), index_col='Report_Ref')
x_testi_10cols = pd.read_csv(os.path.join(proj_root, 'x_testi_10cols.csv'), index_col='Report_Ref')

train_data     = x_train_10cols.to_numpy().astype('float32')
valid_data     = x_valid_10cols.to_numpy().astype('float32')
testi_data     = x_testi_10cols.to_numpy().astype('float32')

#%%
# We'll be looking at multiple loss functions
# One of our loss functions will need data in the range -1/+1 
# i.e., tanh activation, not sigmoid
# but, the data was designed with a set of binary columns (0 or 1)
# to represent whether a given figure is +ve (1) or -ve (0)
# so it expects a sigmoid activation

# so that loss function will require use of slightly different data..
# and we need a conversion function, changing 0 to -1, but leaving 1 as 1

def convert_for_tanh(data):
    # ...get the slice of data we need, ie just the binaries
    bin_cols = data[:,cols_regres: ]
    reg_cols = data[:,0:cols_regres]

    # ...then use 'if 1 then -1 else 1' statement to output -1 for -ve and 1 for +ve
    bin_cols = np.where( bin_cols==0, -1, 1)   

    # concatenate new binary columns with unchanged regression data
    data = np.concatenate((reg_cols, bin_cols), axis=1)

    return data.astype('float32')

train_data = convert_for_tanh(train_data)
valid_data = convert_for_tanh(valid_data)
testi_data = convert_for_tanh(testi_data)

#%%
# some values to get started building our encoder and decoder
# these can be changed later, but its convenient to 
# get an initial version working for the sake of testing
features   = len(x_train_10cols.columns)
alpha      = 0.2 
latent_dim = 2

#%%
# Let's immediately proceed to training the VAE
# We'll consider the sequence to sequence model later, 
# when we have the latent space points which will be the input to that model.
# This VAE will NOT be coded in OOP format.
# Despite being appealing, OOP format with TF2 proved troublesome 
# because not all methods could be inherited, such as model save methods
# This made it burdensome to load and run saved models built using the OOP approach.

def create_encoder(alpha, # leaky alpha to be used in relu activations 
                   latent_dim, # number of latent dimensions
                   layer_nodes_list # list of nodes used in each layer
                   ):

    ### Define VAE encoder ###
    kinit       = 'glorot_normal'

    vae_input_e = Input(shape=(features), dtype='float32')
    for layer_nodes in layer_nodes_list:
        vae_encoded = Dense(layer_nodes, kernel_initializer=kinit)(vae_input_e)
        vae_encoded = LeakyReLU(alpha=alpha)(vae_encoded)

    # latent space, note, no activation of this layer...
    # This is because we will be using tf.nn.sigmoid_cross_entropy_with_logits when calculating the loss
    # There are two params for each latent dimension; z_mean and z_log_var
    vae_latent = Dense(latent_dim*2)(vae_encoded)

    # For the VAE we split the latent space layer into two tensors
    # one represents the mean, the other represents the log of the variance
    z_mean, z_log_var = tf.split(vae_latent, num_or_size_splits = 2, axis = 1)

    # The VAE training loop applies a sampling function next
    # but that is coded into the Gradient Tape training loop.
    # using the function in the next code chunk

    # VAE Encoder Model
    # note output is a list of the mean and the log variance
    vae_encoder = Model(inputs  = vae_input_e, 
                        outputs = [z_mean, z_log_var])

    return vae_encoder

#%%
### SAMPLING THE LATENT SPACE FOR VAE

# VAE needs us to sample the latent space, also known as the 'reparameterization' trick

def reparameterise(z_mean, z_log_var):

    # create a tensor of epsilon (aka perturbation or monte carlo) values 
    # by default we assume the distribution of those random values has mean=0, sd=1
    # tensor shape is same as z_mean
    epsilon = tf.random.normal( shape = z_mean.shape,
                                mean  = 0.0, # default
                                stddev= 1.0, # default
                                dtype = tf.float32)

    # z, the sample from the distribution, is simply mean + a multiple of variance.
    # The multiple is epsilon, which is itself normally distributed
    # result is that we are taking a random normal sample from the distribution of z
    z_sample = z_mean + epsilon * tf.math.exp(z_log_var * 0.5) 

    return z_sample

#%%
# Lets use Eager Execution to test the encoder works as expected...
batch_size_test = 5

## VAE encoder test
# create an encoder
vae_encoder = create_encoder(alpha = alpha, 
                             latent_dim = latent_dim, 
                             layer_nodes_list = [16, 8, 4])

# pass a chunk of data through the encoder
en_test_vae = vae_encoder(x_train_10cols.to_numpy()[0:batch_size_test,:])

# encoder test
# Note, VAE outputs a list; [z_mean, z_log_var]
print("LATENT SPACE")
print("Expecting z_mean and z_log_var each of shape : (", batch_size_test,",", latent_dim,")")
print("  Actual z_mean shape   : (", en_test_vae[0].shape[0], ",", en_test_vae[0].shape[1], ")")
print("  Actual z_log_var shape: (", en_test_vae[1].shape[0], ",", en_test_vae[1].shape[1], ")")

# Now test the sampling of that VAE output
test_reparam = reparameterise(*en_test_vae)
print("AFTER REPARAMETERISATION")
print("Expecting a batch tensor shape of : (", batch_size_test,",", latent_dim,")")
print("  Actual shape :", test_reparam.numpy().shape)

#%%
### Define Decoder ###

def create_decoder(alpha, # leaky alpha for relu activations
                   latent_dim, # number of latent dimensions
                   layer_nodes_list, # list of layer node sizes, one size for each layer
                   activ_bin, # the activation on the binary fields
                   cols_regres=cols_regres, # number of cols of regression data
                   cols_binary=cols_binary # numbe rof cols of binary data
                   ):

    kinit = 'glorot_normal'

    vae_input_d = Input(shape=(None, latent_dim), dtype='float32')

    for layer_nodes in layer_nodes_list:
        vae_decoded = Dense(layer_nodes, kernel_initializer=kinit)(vae_input_d)
        vae_decoded = LeakyReLU(alpha=alpha)(vae_decoded)

    # for the decoder we need to rebuild the output, regression and binary values separately
    # regression values use mse loss so need linear activation (ie none!)
    # binaries can use either sigmoid (0 to 1) or tanh (-1 to +1)
    output_regres = Dense(cols_regres, activation='linear',  kernel_initializer=kinit)(vae_decoded)
    output_binary = Dense(cols_binary, activation=activ_bin, kernel_initializer=kinit)(vae_decoded)

    # Decoder Model
    # Note output is a list so we can apply different losses to these different output types
    vae_decoder = Model(inputs  = vae_input_d, 
                        outputs = [output_regres, output_binary])

    return vae_decoder

#%%
# Lets use Eager Execution to test it...

## VAE decoder test
# create an encoder
vae_decoder = create_decoder(alpha       = alpha, 
                             latent_dim  = latent_dim, 
                             layer_nodes_list = [4, 8, 16],
                             activ_bin   = 'sigmoid',
                             cols_regres = cols_regres, 
                             cols_binary = cols_binary)

# pass a chunk of data through the encoder
de_test_regres_vae, de_test_binary_vae = vae_decoder(test_reparam.numpy())

# decoder test
print("Expecting VAE Output for regression cols with shape: (", batch_size_test,",", cols_regres,")")
print("  Actual Regres= ", de_test_regres_vae.shape)
print()
print("Expecting VAE Output for binary cols with shape: (", batch_size_test,",", cols_binary,")")
print("  Actual Binary= ", de_test_binary_vae.shape)

#%%
### HELPER FOR VAE LOSS FUNCTION
# Note that we're calculating with the log of the variance, instead of the variance, for reasons of numerical stability.

def normal_log_likelihood(z_sample, z_mean, z_log_var, reduce_axis = 1):
    
    loglik = tf.constant(0.5, dtype=tf.float32) * (
                tf.math.log(2 * tf.constant(math.pi, dtype=tf.float32)) +
                z_log_var +
                tf.math.exp(-z_log_var) * tf.square((z_sample - z_mean))
                )

    return -tf.reduce_sum(loglik, axis = reduce_axis)

#%%
### TRAINING LOOP VAE
## Inspired by code (in R) at 
## https://github.com/rstudio/tensorflow-blog/blob/master/_posts/2018-10-22-mmd-vae/mmd_vae.Rmd

# batch size (number of company reports): 100 is preferable for a good sample in a VAE 
# Sample must be big enough for an average to be useful, small enough for learning gradients to be relevant
batch_size = 100

## setup data, epochs, batches and optimiser
num_epochs = 100

# epochs per saving latents and print results to screen
# usually set to 25. Less for testing.
epochs_per_save = 10

def do_train(vae_encoder, 
             vae_decoder,
             filename,
             train_data = train_data,
             valid_data = valid_data,
             recon_type = 'reg_vs_bin', # method for reconstruction loss
             batch_size = 100, 
             num_epochs = 100, 
             epochs_per_save = epochs_per_save,
             save_latents = False):

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    batches_per_epoch = math.ceil(train_data.shape[0]/batch_size)

    # Not yet implemented:
    #   Since we use our own custom training loop we need to implement a validation test each epoch
    #   This would be recorded in the history with a column: 'train_or_valid'. =0 for train, =1 for validation
    
    # prep pandas file to receive training history
    history = pd.DataFrame(columns=['epoch','total_loss',
                                    'recon_loss', 'recon_loss_reg', 'recon_loss_bin', 
                                    'KL_loss'], dtype=float)

    # loop over epochs
    for epoch in range(num_epochs):

        # initialise loss params for each epoch
        loss_total           = 0
        recon_loss_total     = 0
        recon_loss_reg_total = 0
        recon_loss_bin_total = 0
        kl_loss_total        = 0

        # loop over batches in a single epoch
        for batch in range(batches_per_epoch):
        
            # rules for defining batch at start of epoch
            if batch == 0:
                batch_start = 0
                batch_end   = batch_size
            else:
                batch_start = batch_end 
                batch_end   = batch_end + batch_size

            # rules for defining batch at end of epoch
            if batch_start > len(train_data):
                break

            if batch_end > len(train_data):
                batch_end = len(train_data)

            # get batch, all features
            x = train_data[batch_start:batch_end,:]

            # The forward pass is recorded by a GradientTape, 
            # and during the backward pass we explicitly calculate gradients 
            # of the loss with respect to the model’s weights. 
            # These weights are then adjusted by the chosen optimizer.
            with tf.GradientTape(persistent = True) as tape:

            #### RECORD FORWARD PASS ####

                # Encode the batch
                z_mean, z_log_var = vae_encoder( x )

                # Sample latent space
                z_sample = reparameterise(z_mean, z_log_var)

                # Decode the batch of samples
                preds_regres, preds_binary = vae_decoder(z_sample)

                # it will be useful to have a single tensor for all preds
                preds = tf.concat(values=[preds_regres, preds_binary], axis=1)

            #### combined_loss = reconstruction_loss + kl_loss

                # Various loss options are discussed at:
                # https://machinelearningmastery.com/cross-entropy-for-machine-learning/
                # https://gombru.github.io/2018/05/23/cross_entropy_loss/
                    
            # Reconstruction_loss
                if recon_type == 'regression':

                # Recon_Loss OPTION 1
                    # unlogs the data and uses the bin values to give a sign (+ve/-ve) to the value
                    # then uses mse to find difference
                    # the first 2 cols are only ever +ve, named regresA
                    # the next 7 cols (regresB) are pos/neg depending 
                    # on the subsequent binary cols (binaryB).

                    # first slice out the data we need
                    target_regresA = x[:,0:2]
                    target_regresB = x[:,2:cols_regres]
                    target_binaryB = x[:,cols_regres:]

                    preds_regresA  = preds[:,0:2]
                    preds_regresB  = preds[:,2:cols_regres]
                    preds_binaryB  = preds_binary

                    # we leave the data in its log-standardised form
                    # so the loss is not dominated by naturally larger values
                    # BUT we will double the loss wherever the signs are wrong
                    
                    # This is where its convenient to have used tanh activation
                    # and therefore to have converted the data  to -1, +1
                    target_recon = tf.multiply(target_regresB, target_binaryB)
                    preds_recon  = tf.multiply(preds_regresB,  preds_binaryB)

                    # and bring both parts of the data back to one place
                    target_recon = tf.concat(values=[target_regresA, target_recon], axis=1)
                    preds_recon  = tf.concat(values=[preds_regresA,  preds_recon ], axis=1)

                    # finally we can calculate the mean of the square of the error (mse)
                    recon_loss   = tf.reduce_mean(tf.square(preds_recon - target_recon), axis=-1)
                    recon_loss_reg = np.nan
                    recon_loss_bin = np.nan

            # Recon_Loss OPTION 2
                if recon_type == 'reg_vs_bin':
           
                    # this acknowledges that some columns are categorical and others regressional
                    # we then just add the two errors, meaning it remains differentiable

                    # Loss in regression fields as MSE (mean of square of errors)
                    # Returns a scalar loss value per sample (ie per row)
                    recon_loss_reg = tf.reduce_mean(tf.square(preds_regres - x[:,0:cols_regres]), axis=-1)

                    # loss in categorical (1 or 0) fields.
                    # This is a multi class problem 
                    # thus using values which have been sigmoid activated (0 to 1). 
                    # and the target data is also 0, 1
                    # Returns a scalar loss value per sample (ie per row)
                    recon_loss_bin = tf.keras.backend.categorical_crossentropy(
                                        target = x[:,cols_regres:],
                                        output = preds_binary,
                                        axis   = -1,
                                        from_logits = False)
                    
                    # overall reconstruction loss = sum of regression and categorical loss
                    recon_loss = recon_loss_reg + recon_loss_bin

            # Kullback-Leibler Loss
                # loss = y_true * log(y_true / y_pred)
                # Returns a scalar loss value per sample (ie per row)
                kl_loss = tf.keras.losses.KLD(y_true = x, y_pred = preds)
            
                # Simple (unweighted) mean of the the two losses over the batch
                loss = tf.reduce_mean(recon_loss + kl_loss)
                
            # End of Gradient Tape, Forward prop has ended for the batch

            # Sum mean losses for all batches so far 
            # will divide by batch qty at end of epoch to get mean loss per batch
            loss_total       = loss_total       + loss
            recon_loss_total = recon_loss_total + tf.reduce_mean(recon_loss)
            kl_loss_total    = kl_loss_total    + tf.reduce_mean(kl_loss)

            recon_loss_reg_total = recon_loss_reg_total + tf.reduce_mean(recon_loss_reg)
            recon_loss_bin_total = recon_loss_bin_total + tf.reduce_mean(recon_loss_bin)

            # now calculate gradients for encoder
            encoder_gradients  = tape.gradient(loss, vae_encoder.variables)
            # you can then apply your own gradient function, if you wish...
            # encoder_gradients= [process_gradient(g) for g in encoder_gradients]
            grads_and_vars_enc = zip(encoder_gradients, vae_encoder.variables)
            # grads_and_vars is a list of tuples (gradient, variable).
            # Manipulate the 'gradient' part as required, for example cap them, etc.
            # grads_and_vars_enc = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars_enc]

            # apply the gradients to the encoder weights+biases using the optimiser's learning rate      
            optimizer.apply_gradients(grads_and_vars = grads_and_vars_enc,
                                    name = 'apply_encoder_grads')

            # repeat for the decoder...
            decoder_gradients  = tape.gradient(loss, vae_decoder.variables)
            grads_and_vars_dec = zip(decoder_gradients, vae_decoder.variables)
            optimizer.apply_gradients(grads_and_vars = grads_and_vars_dec,
                                    name = 'apply_decoder_grads')

            # end of batch loop
            #print(
            #    "batch=", batch_end, 
            #    ". kl_loss=", tf.reduce_mean(kl_loss).numpy(),
            #    ". recon_loss=", tf.reduce_mean(recon_loss).numpy())

            # Save batch's training results to pandas file
            # losses_batch = [[batch_end, tf.reduce_mean(loss).numpy(), tf.reduce_mean(kl_loss).numpy(), -tf.reduce_mean(recon_loss).numpy()]]
            # history_batch= history_batch.append(pd.DataFrame(data=losses_batch, columns=['batch','loss','kl_loss', 'recon_loss']), ignore_index=True, sort=False)

        # get mean loss per batch (nice to have a big batch size when calculating loss per batch)
        total_loss_perbatch = round(loss_total.numpy()/batches_per_epoch, 10)
        recon_loss_perbatch = round(recon_loss_total.numpy()/batches_per_epoch, 10)
        KL_loss_perbatch    = round(kl_loss_total.numpy()/batches_per_epoch, 10)
        recon_loss_reg_perbatch = round(recon_loss_reg_total.numpy()/batches_per_epoch, 10)
        recon_loss_bin_perbatch = round(recon_loss_bin_total.numpy()/batches_per_epoch, 10)

        # Could execute model on validation data here
        
        # Save epoch's training results to pandas file
        losses_train = [[epoch, total_loss_perbatch, recon_loss_perbatch, recon_loss_reg_perbatch, recon_loss_bin_perbatch, KL_loss_perbatch]]
        history = history.append(pd.DataFrame(data=losses_train, columns=['epoch','total_loss','recon_loss', 'recon_loss_reg', 'recon_loss_bin', 'KL_loss']), ignore_index=True, sort=False)
     
        if epoch % epochs_per_save == 0 or epoch == (num_epochs - 1):

            # save latents every few epochs, so we can plot how they develop
            if save_latents :
            
                # save snapshot of latent space using validation data
                temp_name = os.path.join(proj_root, *subfolders, 'latents_by_epoch', filename+'_latents_epoch'+str(epoch)+'.npy')
                latents   = vae_encoder(valid_data)
                np.save(temp_name, latents)

            # print progress to screen
            print( 'Epoch {}:'.format(epoch),
                ' total: ', total_loss_perbatch,     
                ', recon_loss: ', recon_loss_perbatch,
                ', KL_loss: ', KL_loss_perbatch)
            
        # end of epoch loop
        # history_batch.to_csv(os.path.join(proj_root, *subfolders, filename+'_history_batch.csv'), mode='w', header=True, index=False)

    # save history as csv, all details
    history.to_csv(os.path.join(proj_root, *subfolders, filename+'_history.csv'), mode='w', header=True, index=False)

    # save history as npy, simple epoch and loss only
    np.save(os.path.join(proj_root, *subfolders, filename+'_history.npy'), history[['total_loss']].to_numpy())

    # Return the updated encoder and decoder
    return vae_encoder, vae_decoder, history

#%%
## Save Model Function

from contextlib import redirect_stdout

def save_model_custom(model_object, filename, proj_root=proj_root, subfolders=subfolders):
    
    # save keras model (ie serialize and send to file)
    # This type of save model DOES NOT WORK WITH THE OOP APPROACH. Can you believe it?
    model_object.save(os.path.join(proj_root, *subfolders, filename+'_model.h5'), save_format='tf')

    # save weights
    # So if using the OOP approach, must save weights and keep a copy of the model definition code (above)
    model_object.save_weights(os.path.join(proj_root, *subfolders, filename+'_weights.h5'))

    # save summary text
    filename_txt = os.path.join(proj_root, *subfolders, filename+'_summary.txt')
    with open(filename_txt, 'w') as f:
        with redirect_stdout(f):
            model_object.summary()
    
    # save graph image
    filename_png = os.path.join(proj_root, *subfolders, filename+'_graph.png')
    plot_model(model_object, to_file=filename_png, show_shapes=True)
    
#%%
## GRID SEARCH ##

def do_Grid_Search(params):

    epochs             = params.get('epochs')
    epochs_per_save    = params.get('epochs_per_save')
    encode_layer_nodes = params.get('encode_layer_nodes')
    decode_layer_nodes = params.get('decode_layer_nodes')
    recon_loss_method  = params.get('recon_loss_method')
    leaky_alphas       = params.get('leaky_alphas')
    latent_space_nodes = params.get('latent_space_nodes')
    batch_sizes        = params.get('batch_sizes')

    # prep results table
    results = pd.DataFrame(columns=['encode_layers',   
                                    'decode_layers',    
                                    'recon_type',
                                    'leaky_alpha',
                                    'latent_dim',
                                    'batch_size',
                                    'best_tring',
                                    'best_tring_epc'])
    loop_count = -1

    for encode_layers in encode_layer_nodes :
        for decode_layers in decode_layer_nodes :
            for recon_type in recon_loss_method :
                for leaky_alpha in leaky_alphas :
                    for latent_dim in latent_space_nodes :
                        for batch_size in batch_sizes :
                
                            loop_count += 1
            
                            # filename for model instance
                            filename = ('Loss'   + str(recon_type) + 
                                        '_alpha' + str(leaky_alpha) +
                                        '_latent'+ str(latent_dim) +
                                        '_batch' + str(batch_size))

                            print("loop: ", loop_count,". filename=", filename)
                            print('\n')   

                            # ensure using correct data for selected loss function
                            if recon_type == 'regression':
                                # decoder's activation must be tanh
                                activ_bin = 'tanh'
                                # using tanh, so binaries in data must be -1 to +1
                                t_data = convert_for_tanh(train_data)
                                v_data = convert_for_tanh(valid_data)

                            if recon_type == 'reg_vs_bin':
                                # decoder's activation must be sigmoid
                                activ_bin = 'sigmoid'
                                # data is already in format for sigmoid (0, +1), no action
                                t_data = train_data
                                v_data = valid_data

                            # create encoder and decoder
                            enc = create_encoder(alpha = leaky_alpha, 
                                                latent_dim = latent_dim, 
                                                layer_nodes_list = encode_layers)

                            dec = create_decoder(alpha = leaky_alpha, 
                                                activ_bin   = activ_bin,
                                                latent_dim  = latent_dim, 
                                                cols_regres = cols_regres, 
                                                cols_binary = cols_binary,
                                                layer_nodes_list = decode_layers)
                            
                            # fit model
                            enc, dec, history = do_train(vae_encoder= enc, 
                                                        vae_decoder= dec,
                                                        filename   = filename,
                                                        train_data = t_data,
                                                        valid_data = v_data,
                                                        recon_type = recon_type,
                                                        batch_size = batch_size, 
                                                        num_epochs = epochs,
                                                        epochs_per_save = epochs_per_save)

                            # get result, i.e. min validation error and train error for same epoch
                            best_tring      = min(history['total_loss'])
                            best_tring_epc  = history[history['total_loss'] == best_tring]['epoch']
                            
                            # if unsuccessful training, set to np.nan
                            if len(best_tring_epc) == 0:
                                best_tring_epc = np.nan
                            # else get first item only, in case of tie break.
                            else:
                                best_tring_epc  = best_tring_epc[0:1].item()
                            
                            # save results to table for comparison
                            results.loc[len(results)] = [encode_layers,   
                                                        decode_layers,    
                                                        recon_type,
                                                        leaky_alpha,
                                                        latent_dim,
                                                        batch_size,
                                                        best_tring,
                                                        best_tring_epc]
                            # Save encoder
                            save_model_custom(enc,
                                              filename  = filename+'_encoder',
                                              proj_root = proj_root, 
                                              subfolders= subfolders)
                            
                            # Save decoder
                            save_model_custom(dec,
                                              filename  = filename+'_decoder',
                                              proj_root = proj_root, 
                                              subfolders= subfolders)

                            print(results.loc[len(results)-1])
                            print('\n')

    # save to file because this took a long time (approx 10hrs) to complete!
    results.to_csv(os.path.join(proj_root,*subfolders,'GridScan_Results.csv'))

    return results, 

#%%
# Execute grid search

params =  { 
            'epochs'             : 100 ,
            'epochs_per_save'    : 25 ,
            'encode_layer_nodes' : [[16, 8, 4]] ,
            'decode_layer_nodes' : [[ 4, 8,16]] ,
            'recon_loss_method'  : ['regression', 'reg_vs_bin'] ,
            'leaky_alphas'       : [0.0, 0.2, 0.4] ,
            'latent_space_nodes' : [2, 3] ,
            'batch_sizes'        : [100]
           } 

results = do_Grid_Search(params=params)

# View results
results

#%%

# For both loss methods the lowest losses were encountered with:
# Alpha = 0.0 (i.e. simple ReLu)
# Latent_Dim = 3. 
# It is expected that 3 would be better than 2. However, 

# we'll train the model to 400 epochs and view the training history
# to seek any overtraining

params =  { 
            'epochs'             : 400 ,
            'epochs_per_save'    : 10 ,
            'encode_layer_nodes' : [[16, 8, 4]] ,
            'decode_layer_nodes' : [[ 4, 8,16]] ,
            'recon_loss_method'  : ['regression'] ,
            'leaky_alphas'       : [0.0] ,
            'latent_space_nodes' : [2, 3] ,
            'batch_sizes'        : [100]
           } 

results = do_Grid_Search(params=params)

# View results
results


#%%
# Explore Results of training

filename_enc ='Lossregression_alpha0.0_latent2_batch100_decoder_model.h5'
filename_dec ='Lossregression_alpha0.0_latent2_batch100_encoder_model.h5'

# Load the model, because this chunk will follow a few hours after the training above
vae_encoder = load_model(os.path.join(proj_root, *subfolders, filename_enc), compile=False)
vae_decoder = load_model(os.path.join(proj_root, *subfolders, filename_dec), compile=False)

#%%
############ FUNCTION TO PLOT ALL TRAINING HISTORIES OR ALL EMBEDDINGS ##################################

import matplotlib.pyplot as plt

## Function to get list of folders in parent folder
def get_list_of_folders(parent_folder_name):
    mypath = os.path.join(parent_folder_name)
    folders_list = [f for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]
    folders_list_df = pd.DataFrame(folders_list, columns=['foldername'])
    folders_list_df['progress'] = 'Pending'
    del folders_list
    return folders_list_df

## Function to get list of all files in a given folder
def get_list_files_in_folder(folder_name):
    mypath = os.path.join(proj_root, folder_name)
    files_list = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    files_list_df = pd.DataFrame(files_list, columns=['filename'])
    files_list_df['progress'] = 'Pending'
    del files_list
    return files_list_df


#%%

## Function to plot all histories found in a folder of saved histories

def matrix_of_plots(embed_data     = valid_data,          # test data in pandas
                    data_name_5char= 'SEC  ',       # a name for the chart title
                    plot_type      = 'histories',   # 'embeddings' or 'histories'
                    charts_per_col = 2              # must be at least 2, else code errors
                    ):

    # get all files
    rootfiles    = get_list_files_in_folder(os.path.join(proj_root, *subfolders))['filename']
    latentfiles  = get_list_files_in_folder(os.path.join(proj_root, *subfolders, 'latents_by_epoch'))['filename']

    # filter down to training histories or embeddings
    if plot_type == 'histories':
        chartfiles = [file for file in rootfiles if file.endswith('_history.npy') ]
        prefix = 'Training histories for '
        xlabel = 'epochs'
        ylabel = 'loss'

    else: # embeddings
        chartfiles = [file for file in latentfiles if file.endswith('.npy') ]
        prefix = 'Embeddings for '
        xlabel = 'latent x'
        ylabel = 'latent y'

    # how many rows of training figures? Never less than 1
    rows = max(len(chartfiles) // charts_per_col, 1)
    cols = max(len(chartfiles) // rows, 1)

    # set size of matplotlib canvas
    plt.figure(num=None, figsize=(10, 10)) #dpi=80, facecolor='w', edgecolor='k'

    # set up the canvas
    canvas, figures = plt.subplots(rows, cols)

    # must ensure figures is a 2D numpy array
    if type(figures) != np.ndarray:
        figures = np.array(figures)

    assert type(figures) == np.ndarray, "error conversion to numpy array"

    dims = len(figures.shape)
    while dims < 2 :
        figures = np.expand_dims(figures, axis=0)
        dims = len(figures.shape)

    # initialise row and column indices
    row = 0
    col = 0

    # loop around models, plotting each onto a subplot on the canvas
    for file_name in chartfiles:

        # make title

        epoch_start = file_name.find('epoch')
        epoch_end   = file_name.find('.npy')
        title = file_name[epoch_start:epoch_end]

        # setup subplot and title
        figures[row,col].set_title(title, size=7)
        figures[row,col].tick_params(labelsize=6)

        # space is limited, so only add axis labels at far left and foot of canvas, but not onto every subplot
        if col == 0:
            figures[row,col].set_ylabel(ylabel, size=5)
        if row == rows-1:
            figures[row,col].set_xlabel(xlabel, size=5)
        
        if plot_type == 'embeddings':

            # load latents for epoch
            z_vectors = np.load(os.path.join(proj_root,*subfolders,'latents_by_epoch', 'VAE_TS_SEC_latents_epoch0.npy'))

            # note, when we apply the encoder to a data set, eg validation
            # then we get numpy array shape (2, samples, 2)
            # shape[0] is [z_mean, z_log_var]
            # shape[1] is number of records (aka samples) in set
            # shape[2] is the latent dimensions, 2 in this case
            # for embeddings we don't want z_log_var, only z_mean
            z_vectors = z_vectors[0,:,:]

            # convert test embeddings to pandas
            z_vectors = pd.DataFrame(data=z_vectors, columns=['x','y'])

            # let's see if we can get a colour from the Turnover...
            z_vectors['StockholdersEquity'] = embed_data['StockholdersEquity'].reset_index(drop=True)

            # plot it
            figures[row,col].scatter(x=z_vectors['x'], y=z_vectors['y'], c=z_vectors['StockholdersEquity'])

        else: # training history

            # load data
            tng_history = np.load(os.path.join(proj_root,*subfolders,file_name))

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
    plt.savefig(os.path.join(proj_root, *subfolders, prefix + data_name_5char + ' data.jpg'),  dpi=300)

#%%
############ INSPECT TRAINING #####################

matrix_of_plots(embed_data=x_valid_10cols, plot_type='histories')

#%%
############ INSPECT EMBEDDINGS (LATENTS) #####################
# this displays the embeddings for each epoch, showing how they develop over time
# colours represent business size via 'StockholdersEquity'

matrix_of_plots(embed_data=x_valid_10cols, plot_type='embeddings')

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
    inputs  = np.array(x_train_10cols.sample(32))
    # get outputs, there are two (regression and binary) so concatenate them using hstack
    outputs = np.hstack(autoencoder.predict(inputs))
    recon_errs_real = get_reconstruction_losses(inputs  = inputs,outputs = outputs)
    
    # Get Reconstruction Errors for fake data
    inputs  = generate_fake_samples_fromRandoms(dataset = x_train_10cols,qty = 32)
    # get outputs, there are two (regression and binary) so concatenate them using hstack
    outputs = np.hstack(autoencoder.predict(inputs))
    recon_errs_fake = get_reconstruction_losses(inputs  = inputs,outputs = outputs)

    recon_errs_real_mean.append(np.mean(recon_errs_real))
    recon_errs_fake_mean.append(np.mean(recon_errs_fake))

## Plot mean reconstruction errors as overlaid histograms

bins = np.linspace(0, 1.5, 50)
plt.hist(recon_errs_real_mean, bins, alpha=0.5, label='real')
plt.hist(recon_errs_fake_mean, bins, alpha=0.5, label='fake')
plt.legend(loc='upper right')
plt.title('Mean Error in 150 Fake vs Real Reconstructions')
plt.show()

#%%
# RESULT
# The lower the error the better, happily the fake and real errors distributions don't overlap!
# If the mean error in a batch is around 0.6 then data is indistinguishable from real
# If the mean error in a batch is around 1.0 then data is indistinguishable from fake
# Any batch with a mean error between 0.6 and 1.0 is a mix of fake and real reports
# This is a useful scoring mechanism for our VAE and GAN

print("Mean error in batch of real data: ", np.mean(np.array(recon_errs_real_mean)) )
print("Mean error in batch of fake data: ", np.mean(np.array(recon_errs_fake_mean)) )
