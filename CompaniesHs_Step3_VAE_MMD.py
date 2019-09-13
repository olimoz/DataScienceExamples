### INTRODUCTION

# The MMD-VAE is a type of variational autoencoder (VAE), 
# but the latent space prior is not a simple gaussian
# instead it may be mix of gaussians, permitting a 'hilly' landscape
# to be learned within the latent space
# This allows the data to find its own distribution, 
# which leads to sharper edges in visual models

# The loss function which accompanies this is marginally different to the VAE
# The VAE loss function is the 'Evidence Lower Bound', ie ELBO:
#   ELBO = Reconstruction Loss - Kullback Leibler Divergence
#       where KL Divergence is a -ve number
#       KL divergence measure the error in approximating a real distribution with a simpilifed one, such as a gaussian
#       Basic autoencoders measure only their reconstruction loss

# The first component is the reconstruction loss which is the only loss used in plain
# (non-variational) autoencoders. 
# The second component is the Kullback-Leibler divergence between a prior imposed on the latent 
# space (typically, a standard normal distribution) and the representation of latent 
# space as learned from the data.

# Imposing a Gaussian on the latent space is somewhat arbitrary, enter the MMD:
# This is a subtype of the Info-VAE that instead of making each representation in 
# latent space as similar as possible to the prior, coerces the respective distributions
# to be as close as possible. MMD stands for maximum mean discrepancy, 
# a similarity measure for distributions based on matching their respective moments.

# This code was inspired by MMD VAE code which can be found in R at RStudio's blog
#   https://blogs.rstudio.com/tensorflow/posts/2018-10-22-mmd-vae/
#   This is in R, but translating from R to Python is useful for learning! Can't just copy/paste

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#%%
#### IMPORTS
import os
import pandas as pd
import numpy as np
import math as math
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization, Concatenate, Reshape
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import plot_model
from tensorflow_probability import distributions as tfd
import cProfile

#%% 
#### Confirm environment

print(tf.__version__)
tf.executing_eagerly()

#%%
### LOAD DATA

proj_root  = 'E:\\Data\\CompaniesHouse'
subfolders = ['SaveModels_MMDVAE']

prepped_10cols = pd.read_csv(os.path.join(proj_root, 'prepped_10cols.csv'))
sds_10cols     = pd.read_csv(os.path.join(proj_root, 'sds_10cols.csv'), index_col=0)
means_10cols   = pd.read_csv(os.path.join(proj_root, 'means_10cols.csv'), index_col=0)

x_train_10cols = pd.read_csv(os.path.join(proj_root, 'x_train_10cols.csv'), index_col='Report_Ref')
x_train_10cols = x_train_10cols.drop(columns=['Unnamed: 0'])

x_valid_10cols = pd.read_csv(os.path.join(proj_root, 'x_valid_10cols.csv'), index_col='Report_Ref')
x_valid_10cols = x_valid_10cols.drop(columns=['Unnamed: 0'])

x_testi_10cols = pd.read_csv(os.path.join(proj_root, 'x_testi_10cols.csv'), index_col='Report_Ref')
x_testi_10cols = x_testi_10cols.drop(columns=['Unnamed: 0'])

#%%
# Each model could use any of the data sets
# To enable easily reusable code, we define which dataset will be used...

train_data = x_train_10cols
testi_data = x_testi_10cols
valid_data = x_valid_10cols


#%%
############ ENCODER #################

# Coded in OOP format as a custom tensorflow model, as opposed to static functions
# This coding approach is 'pythonic' and reminiscent of pytorch
# BUT
# a) its harder to debug AND in subclassed models, 
# b) the model's topology is defined as Python code (rather than as a static graph of layers). 
# That means the model's topology cannot be inspected or serialized. 
# As a result, the following methods and attributes are not available for subclassed models:
# model.inputs, model.outputs, model.to_yaml(), model.to_json(), model.get_config() and model.save()

class encoder_mdl(tf.keras.Model):

    def __init__(self, params_dict, name, **kwargs):

        super(encoder_mdl, self).__init__(name=name, **kwargs)

        # Get params
        self.original_dim       = params_dict.get('original_dim')
        self.layer_nodes_list   = params_dict.get('layer_nodes_list') # is list, not single object
        self.latent_space_dims  = params_dict.get('latent_space_dims')
        self.kinit              = params_dict.get('kinit')
        self.leaky_alpha        = params_dict.get('leaky_alpha')
        self.apply_batchnorm    = params_dict.get('apply_batchnorm')

        self.input_layer        = Input(shape=(22,), name='encoder_input')

        # define possible encoder layers, there can be any number, 
        # eg 3 layers: 16 nodes, 8 nodes, 2 nodes. Or only 2 layers: 32 nodes, 16 nodes.
        # Therefore, we use exec() to define the class variables dynamically
        for layer_nodes in self.layer_nodes_list :
            i = self.layer_nodes_list.index(layer_nodes)
            
            # define dense layer
            exec("self.enc_lyr_{} = Dense(layer_nodes, kernel_initializer=self.kinit, name='enc_lyr_'+str({}))".format(i, i))
            
            # define leaky activation for above dense layer
            exec("self.enc_lyr_{}_lky = LeakyReLU(alpha=self.leaky_alpha, name='enc_lyr_'+str({})+'_lky')".format(i, i))

        # define a batch norm layer
        if self.apply_batchnorm:
            self.enc_batchnorm = BatchNormalization()

        # define final dense layer for input to latent space
        self.latent_dense     = Dense(self.latent_space_dims, name='latent_dense')
        self.latent_dense_lky = LeakyReLU(alpha=self.leaky_alpha, name='latent_dense_lky')
        
    def call(self, inputs, dynamic_creation=False):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        encoded_0 = inputs

        k = len(self.layer_nodes_list)

        if not dynamic_creation :
            # hard coded to three dense layers, their activation and batchnorm, if req'd
            encoded = self.enc_lyr_0(encoded_0)
            if self.apply_batchnorm:
                encoded = self.enc_batchnorm(encoded)
            encoded = self.enc_lyr_0_lky(encoded)
            
            encoded = self.enc_lyr_1(encoded)
            if self.apply_batchnorm:
                encoded = self.enc_batchnorm(encoded)
            encoded = self.enc_lyr_1_lky(encoded)        
            
            encoded = self.enc_lyr_2(encoded)
            if self.apply_batchnorm:
                encoded = self.enc_batchnorm(encoded)
            encoded = self.enc_lyr_2_lky(encoded)
            
            encoded = self.latent_dense(encoded)
            if self.apply_batchnorm:
                encoded = self.enc_batchnorm(encoded)
            output  = self.latent_dense_lky(encoded)

            return output

        # The below code has been left in place in order to document
        # an approach to dynamically creating any number of layers in a custom keras model
        # No errors generated and model.summary() returns the expected layers and dims.
        # BUT the model does not process the input, 'None' object is outputted. Cannot be used.
        else:
        
            for layer_nodes in self.layer_nodes_list:

                i = self.layer_nodes_list.index(layer_nodes)

                # define dense looped layer, can be a varying number of these, so use exec()
                # optionally apply batch norm
                if self.apply_batchnorm:
                    exec('encoded_{}_de = self.enc_lyr_{}(encoded_{})'.format(i,i,i))
                    exec('encoded_{}_bn = self.enc_batchnorm(encoded_{}_de)'.format(i,i))
                    exec('encoded_{}    = self.enc_lyr_{}_lky(encoded_{}_bn)'.format(i+1,i,i))

                else:
                    exec('encoded_{}_de = self.enc_lyr_{}(encoded_{})'.format(i,i,i))
                    exec('encoded_{}    = self.enc_lyr_{}_lky(encoded_{}_de)'.format(i+1,i,i))

            # final layer enforces latent space dims
            # must use 'exec' because these layers reference variables 
            exec('encoded_{} = self.latent_dense(encoded_{})'.format(k+1, k))
            output = exec('self.latent_dense_lky(encoded_{})'.format(k+1))

            return output

    # Need this method to enable serializable (ie save model)
    def get_config(self):
        config = super(encoder_mdl, self).get_config()
        config.update({'params_dict': self.params_dict})
        return config

    # Also, need this method to enable serializable (ie save model)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = [self.latent_space_dims,]
        return tf.TensorShape(shape)

#%%
## Let's test it!

# set params
params_dict =  { 
                'original_dim'       : 22,
                'layer_nodes_list'   : [16, 8, 4], # MUST BE THREE LAYERS. HARD CODED
                'latent_space_dims'  : 2,
                'kinit'              : 'glorot_normal',
                'leaky_alpha'        : 0.2,
                'apply_batchnorm'    : False,
                }

# instantiate the new model class
encoder = encoder_mdl(name='my_encoder', params_dict=params_dict)

# execute it with a few lines of training data
# Note np.array is fed to inputs, NOT pandas table
test_encoder = encoder(np.array(train_data[0:5]))

# We're using eager execution, so let's inspect the ouput
# Is the shape right? Expecting [5,2]
test_encoder.shape

#%%
# Let's test it!  cont...

# now let's view a summary, see if its as expected
encoder.summary()

#%%
################ DECODE ##################
# Note, the decoder does not output logits for each field
# instead it outputs probabilities for each field
# This is done using Tensorflow Probability
# See https://blogs.rstudio.com/tensorflow/posts/2019-01-08-getting-started-with-tf-probability/

class decoder_mdl(tf.keras.Model):

    def __init__(self, params_dict, name, **kwargs):
        super(decoder_mdl, self).__init__(name=name, **kwargs)

        # Get params
        self.original_dim       = params_dict.get('original_dim')
        self.layer_nodes_list   = params_dict.get('layer_nodes_list')[::-1] # reversed, cos params originally intended for encoder
        self.latent_space_dims  = params_dict.get('latent_space_dims')
        self.kinit              = params_dict.get('kinit')
        self.leaky_alpha        = params_dict.get('leaky_alpha')
        self.apply_batchnorm    = params_dict.get('apply_batchnorm')

        # define final dense layer for input to latent space
        self.latent_dense     = Dense(self.latent_space_dims, name='latent_dense')
        self.latent_dense_lky = LeakyReLU(alpha=self.leaky_alpha, name='latent_dense_lky')
        
        # define a batch norm layer
        if self.apply_batchnorm:
            self.dec_batchnorm = BatchNormalization()

        # define possible decoder layers, there can be any number, 
        # eg 3 layers: 4 nodes, 8 nodes, 16 nodes. Or only 2 layers: 5 nodes, 10 nodes.
        # Therefore, we use exec() to define the class variables dynamically
        for layer_nodes in self.layer_nodes_list :
            i = self.layer_nodes_list.index(layer_nodes)

            # define dense layer
            exec("self.dec_lyr_{} = Dense(layer_nodes, kernel_initializer=self.kinit, name='dec_lyr_'+str({})+'_nodes'+str({}))".format(i, i, layer_nodes))
            
            # define leaky activation for above dense layer
            exec("self.dec_lyr_{}_lky = LeakyReLU(alpha=self.leaky_alpha, name='dec_lyr_'+str({})+'_lky_nodes'+str({}))".format(i, i, layer_nodes))

        # define reconstruction layer(s)
        # define layer to reconstitute the values (fields 1 to 12)
        self.recon_regres = Dense(12, kernel_initializer=self.kinit, activation='linear',  name='recon_regres')
        
        # define layer to reconstitute a binary, uses sigmoid activation (fields 13 to 22)
        self.recon_binary = Dense(10, kernel_initializer=self.kinit, activation='sigmoid', name='recon_binary')

        # concatenate before presentation
        self.reconstruction = Concatenate()

    def call(self, latent, dynamic_creation=False):

        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        decoded_0 = latent

        decoded_0 = self.latent_dense(decoded_0)
        decoded_0 = self.latent_dense_lky(decoded_0)

        k = len(self.layer_nodes_list)

        if not dynamic_creation :
            # hard coded to three dense layers, their activation and batchnorm, if req'd
            decoded = self.dec_lyr_0(decoded_0)
            if self.apply_batchnorm:
                decoded = self.dec_batchnorm(decoded)
            decoded = self.dec_lyr_0_lky(decoded)
            
            decoded = self.dec_lyr_1(decoded)
            if self.apply_batchnorm:
                decoded = self.dec_batchnorm(decoded)
            decoded = self.dec_lyr_1_lky(decoded)        
            
            decoded = self.dec_lyr_2(decoded)
            if self.apply_batchnorm:
                decoded = self.dec_batchnorm(decoded)
            decoded = self.dec_lyr_2_lky(decoded)

            # There are two outputs, values for regression (mse)...
            output_regres = self.recon_regres(decoded)

            # ...and binaries for cross entropy
            output_binary = self.recon_binary(decoded)

            # output
            output = self.reconstruction([output_regres, output_binary])

            return output

        # The below code has been left in place in order to document
        # an approach to dynamically creating any number of layers in a custom keras model
        # No errors generated and model.summary() returns the expected layers and dims.
        # BUT the model does not process the input, 'None' object is outputted. Cannot be used.
        else:
            # define possible decoder layers, there can be any number, 
            # eg 3 layers: 4 nodes, 8 nodes, 16 nodes. Or only 2 layers: 5 nodes, 10 nodes.
            # Therefore, we use exec() to define the class variables dynamically          
            for layer_nodes in self.layer_nodes_list:

                i = self.layer_nodes_list.index(layer_nodes)

                # define dense looped layer, can be a varying number of these, so use exec()
                # optionally apply batch norm
                if self.apply_batchnorm:
                    exec('decoded_{}_de = self.dec_lyr_{}(decoded_{})'.format(i,i,i))
                    exec('decoded_{}_bn = self.dec_batchnorm(decoded_{}_de)'.format(i,i))
                    exec('decoded_{}    = self.dec_lyr_{}_lky(decoded_{}_bn)'.format(i+1,i,i))

                else:
                    exec('decoded_{}_de = self.dec_lyr_{}(decoded_{})'.format(i,i,i))
                    exec('decoded_{}    = self.dec_lyr_{}_lky(decoded_{}_de)'.format(i+1,i,i))

            # There are two outputs, values for regression (mse)...
            # must use 'exec' because these layers reference variables only declared in exec() above
            exec('output_regres = self.recon_regres(decoded_{})'.format(k+1))

            # ...and binaries for cross entropy
            exec('output_binary = self.recon_binary(decoded_{})'.format(k+1))

            # concatenate so there is only one output object
            output = exec('self.reconstruction([output_regres, output_binary])')

            return output

    # Need this method to enable serializable (ie save model)
    def get_config(self):
        config = super(decoder_mdl, self).get_config()
        config.update({'params_dict': self.params_dict})
        return config

    # Also, need this method to enable serializable (ie save model)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = [self.original_dim,]
        return tf.TensorShape(shape)

#%%
### Test it!

# instantiate the new model class
decoder = decoder_mdl(name='my_decoder', params_dict=params_dict)

# execute it with a few lines of training data
# Note np.array is fed to inputs, NOT pandas table
test_decoder = decoder(test_encoder)

# We're using eager execution, so let's inspect the output
# Is the shape right? Expecting [5,22]
test_decoder.shape

#%%
### Test it! cont...

# now let's view a summary, see if its as expected
decoder.summary()

#%%
### HELPERS FOR MMD LOSS FUNCTION

# The loss, maximum mean discrepancy (MMD), is based on the idea that
# two distributions are identical if and only if all moments are identical. 
# Concretely, MMD is estimated using a kernel, such as the Gaussian kernel 
# to assess similarity between distributions.

#   k(z,z′)=e^(||z−z′||)/2σ2

# The idea then is that if two distributions are identical, 
# the average similarity between samples from each distribution should be identical
# to the average similarity between mixed samples from both distributions

# below code taken directly from the original author of the MMD
# https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.py

def compute_kernel(x, y):
    x_size  = tf.shape(x)[0]
    y_size  = tf.shape(y)[0]
    dim     = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel  = compute_kernel(x, x)
    y_kernel  = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


#%%
## Save Model Function

from contextlib import redirect_stdout

def save_model_custom(model_object, filename, proj_root=proj_root, subfolders=subfolders):
    
    # save keras model (ie serialize and send to file)
    # Saving the model to HDF5 format does not work for subclassed models, 
    # because such models are defined via the body of a Python method, which isn't safely serializable. 
    # COMMENTED OUT
    # model_object.save(os.path.join(proj_root,model_root,filename+'_model.h5'), save_format='tf')

    # save weights
    model_object.save_weights(os.path.join(proj_root, *subfolders, filename+'_weights.h5'))

    # save summary text
    filename_txt = os.path.join(proj_root, *subfolders, filename+'_summary.txt')
    with open(filename_txt, 'w') as f:
        with redirect_stdout(f):
            model_object.summary()
    
    # save graph image
    # COMMENTED OUT
    # plot_model does not work properly for subclassed models either. Hence commented out
    # filename_png = os.path.join(proj_root,model_root,filename+'_graph.png')
    # plot_model(model_object, to_file=os.path.join(proj_root,model_root,filename_png), show_shapes=True)
   
    # save training history
    # COMMENTED OUT - Train loop saves history
    #if save_history:
    #   filename_history = os.path.join(proj_root,model_root,filename+'_history.npy')
    #   l_loss  = np.array(history.history['loss'])
    #   l_loss  = np.reshape(l_loss, (l_loss.shape[0],1))
    #   l_vloss = np.array(history.history['val_loss'])
    #   l_vloss = np.reshape(l_vloss, (l_vloss.shape[0],1))
    #   np.save(file = filename_history, arr = np.concatenate((l_loss, l_vloss), axis=1))

#%%
### SET UP THE TRAINING LOOP
params_dict =  { 
                'original_dim'       : 22,
                'layer_nodes_list'   : [16, 8, 4], # MUST BE THREE LAYERS. HARD CODED
                'latent_space_dims'  : 2,
                'kinit'              : 'glorot_normal',
                'leaky_alpha'        : 0.2,
                'apply_batchnorm'    : False,
                }

# setup data, epochs, batches and optimiser

num_epochs = 600
batch_size = 100 # save as VAE for similar sample size reasons
optimizer  = tf.keras.optimizers.Adam(1e-4)
filename   = 'data_10col_lossfn_2types_alpha_'+str(params_dict.get('leaky_alpha'))+\
              '_norm_'+str(params_dict.get('apply_batchnorm'))+\
              '_batch_'+str(batch_size)
latent_space_dims = 2
batches_per_epoch = math.ceil(len(train_data)/batch_size)

# instantiate encoder and decoder
encoder = encoder_mdl(name='mmd_encoder', params_dict=params_dict)
decoder = decoder_mdl(name='mmd_decoder', params_dict=params_dict)

# prep pandas file to receive training history. 
# Not yet implemented: 
#   Since we use our own custom training loop we need to implement a validation test each epoch
#   This would be recorded in the history with a column: 'train_or_valid'. =0 for train, =1 for validation
history = pd.DataFrame(columns=['epoch','total_loss','loss_nll_total','loss_mmd_total'], dtype=float)

#%%
### TRAINING LOOP

# loop over epochs
for epoch in range(num_epochs):

    # initialise loss params for each epoch
    total_loss     = 0
    loss_nll_total = 0
    loss_mmd_total = 0

    # loop over batches in a single epoch
    for batch in range(batches_per_epoch):
    
        # rules for start of epoch
        if batch == 0:
            batch_start = 0
            batch_end   = batch_size
        else:
            batch_start = batch_end 
            batch_end   = batch_end + batch_size

        # rules for end of epoch
        if batch_start > len(train_data):
            break

        if batch_end > len(train_data):
            batch_end = len(train_data)

        # get batch as numpy
        x = np.array(train_data)[batch_start:batch_end,]
        
        # The forward pass is recorded by a GradientTape, 
        # and during the backward pass we explicitly calculate gradients 
        # of the loss with respect to the model’s weights. 
        # These weights are then adjusted by the optimizer.
        with tf.GradientTape(persistent = True) as tape:

        #### RECORD FORWARD PASS ###
            mean  = encoder(x)
            preds = decoder(mean)

            # Could forward pass validation batch here

        ### CALCULATE MMD ###

            # generate some random gaussians to compare with real values 
            true_samples = tf.random.normal(
                                shape = [batch_size, latent_space_dims],
                                mean  = 0.0,
                                stddev= 1.0,
                                dtype = tf.float32)
            
            # compute the MMD distance between the random Gaussians and real samples
            loss_mmd = compute_mmd(true_samples, mean)

            # compute a reconstruction loss
            loss_nll = tf.reduce_mean(tf.square(x - preds))

            # batch loss = reconstruction loss + MMD loss
            loss = loss_nll + loss_mmd
        
        # End of Gradient Tape, Forward prop has ended for the batch

        # Sum losses for batches so far 
        # will divide by batch qty at end of epoch to get mean loss per batch
        total_loss     = loss     + total_loss
        loss_mmd_total = loss_mmd + loss_mmd_total
        loss_nll_total = loss_nll + loss_nll_total
        
        # now calculate gradients for encoder
        encoder_gradients = tape.gradient(loss, encoder.variables)
        
        # and gradients for decoder
        decoder_gradients = tape.gradient(loss, decoder.variables)
        
        # apply the gradients to the ENCODER weights+biases using the optimiser's learning rate
        grads_and_vars_enc = list(zip(encoder_gradients, encoder.variables))
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_enc,
                                  name = 'apply_encoder_grads')

        # apply the gradients to the DECODER weights+biases using the optimiser's learning rate
        grads_and_vars_dec = list(zip(decoder_gradients, decoder.variables))
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_dec,
                                  name = 'apply_decoder_grads')
    
    # end of batch loop

    # get mean loss per batch (nice to have a big batch size when calculating loss per batch)
    total_loss_perbatch     = round(    total_loss.numpy()/batches_per_epoch, 10)
    loss_nll_total_perbatch = round(loss_nll_total.numpy()/batches_per_epoch, 10)
    loss_mmd_total_perbatch = round(loss_mmd_total.numpy()/batches_per_epoch, 10)
   
    # Save epoch's training results to pandas file
    losses_train = [[epoch, total_loss_perbatch, loss_nll_total_perbatch, loss_mmd_total_perbatch]]
    history = history.append(pd.DataFrame(data=losses_train, columns=['epoch','total_loss','loss_nll_total','loss_mmd_total']), ignore_index=True)

    # every few epochs
    if epoch % 10 == 0 or epoch == num_epochs: #every tenth epoch

        # save snapshot of latent space using validation data
        temp_name = os.path.join(proj_root, *subfolders, 'latents_by_epoch', filename+'_latents_epoch'+str(epoch)+'.npy')
        latents   = encoder(np.array(valid_data))
        np.save(temp_name, latents)

        # print losses to screen
        print( 'Epoch {}:'.format(epoch),
            ' total: ',          total_loss_perbatch,     
           ', loss_nll_total: ', loss_nll_total_perbatch,
           ', loss_mmd_total: ', loss_mmd_total_perbatch)
        
    # end of epoch loop
    
#%%
# Save models, weights and training history

# Subclassed models cannot be saved in tf.keras because
# such models are defined via the body of a Python method, which isn't safely serializable 
# So we can only save the weights
history.to_csv(os.path.join(proj_root, 'SaveModels_MMDVAE', filename+'_history.csv'), mode='w', header=True, index=False)

# Save encoder
save_model_custom(model_object = encoder,
                  filename     = filename+'_enc')

# save decoder
save_model_custom(model_object = decoder,
                  filename     = filename+'_dec')

#%%
#### INSPECT TRAINING CURVE

#%%

import matplotlib.pyplot as plt

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
    files_list = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    files_list_df = pd.DataFrame(files_list, columns=['filename'])
    files_list_df['progress'] = 'Pending'
    del files_list
    #print(len(files_list_df.index))
    return files_list_df

#%%
## Function to plot all histories found in a folder of saved histories
def matrix_of_plots(
                    data_name_5char= '10col',      # '10col' or 'catego'
                    test_data_df   = testi_data,        # test data in pandas
                    plot_type      = 'embeddings', # 'embeddings' or 'histories'
                    subfolders     = ['SaveModels_MMDVAE', 'latents_by_epoch'],
                    charts_per_col = 5, # must be at least 2, else code errors
                    figsize_x      = 10,
                    figsize_y      = 20,
                    dpi            = 100,
                    many_plots_on_one_canvas = False
                    ):

    # get all files
    save_models = get_list_files_in_folder(os.path.join(proj_root, *subfolders))['filename']
    save_models = [file for file in save_models if file.endswith('.npy') and file.startswith('data_' + data_name_5char)]

    def instantiate_plot(rows, cols):
        # set up the canvas
        canvas, figures = plt.subplots(rows, cols, figsize=(figsize_x, figsize_y), dpi=dpi)
        
        # if only one row and one column, force into numpy array of one row and one column
        if not type(figures) == np.ndarray:
            figures = np.array([[figures]], dtype=object)

        # if only one row, then need to force row and column ref
        if len(figures.shape) == 1:
            figures = figures.reshape(1, figures.shape[0])
        
        return(canvas, figures)

    def finish_plot(canvas, suptitle, savefilename):

        # try to prevent overlap between subplot titles, axis labels and ticks
        canvas.tight_layout()

        # add title
        canvas.suptitle(suptitle, fontsize=11)

        # ensure canvas title does not overlap plots
        canvas.subplots_adjust(top=0.88)

        # save result
        plt.savefig(os.path.join(proj_root, *subfolders, savefilename))

    if many_plots_on_one_canvas:
        # how many rows of training figures?
        rows = len(save_models) // charts_per_col
        cols = len(save_models) // rows

        #instantiate one big canvas for all figures
        canvas, figures = instantiate_plot(rows, cols)

        # set text sizes
        size_axislabel = 5
        size_axisdigit = 6 
        size_title     = 7
    else:
        # only one plot on each canvas, we save many canvases!
        rows = 1
        cols = 1

        # set text sizes
        size_axislabel = figsize_x*2.5
        size_axisdigit = figsize_x*2
        size_title     = figsize_x*4

        # DONT instantiate canvas and figures just yet....

    # initialise row and column indices
    row = 0
    col = 0

    # loop around models, plotting each onto a subplot on the canvas
    for file_name in save_models:

        if not many_plots_on_one_canvas:
            # instantiate one canvas for one figure
            canvas, figures = instantiate_plot(rows, cols)

        figures[row,col].tick_params(labelsize=size_axisdigit)

        # space is limited, so only add axis labels at far left and foot of canvas, but not onto every subplot
        if plot_type == 'embeddings':

            # set title
            epoch_locn = file_name.find('epoch')
            epoch_num  = file_name[epoch_locn+5:len(file_name)-4]
            title      = 'Embeddings Epoch '+str(epoch_num)
            figures[row,col].set_title(title, size=size_title)

            # set axis labels
            if col == 0:
                figures[row,col].set_ylabel('latent x', size=size_axislabel)
            if row == rows-1:
                figures[row,col].set_xlabel('latent y', size=size_axislabel)

            # load numpy file
            z_vectors = np.load(os.path.join(os.path.join(proj_root, *subfolders),file_name))

            # convert test embeddings to pandas
            z_vectors = pd.DataFrame(data=z_vectors, columns=['x','y'])

            # let's see if we can get a colour from the Turnover...
            z_vectors['turnover_curr'] = test_data_df['001_SOI_TurnoverRevenue_curr'].reset_index(drop=True)

            # plot it
            figures[row,col].scatter(x=z_vectors['x'], y=z_vectors['y'], c=z_vectors['turnover_curr'])

            #savefilename
            savefilename = 'Plot_'+plot_type + '_'+ data_name_5char + '_epoch_'+ epoch_num +'.jpg'
            
        else: # training history

            # load data
            tng_history = np.load(os.path.join(proj_root,*subfolders,file_name))

            # set title
            epoch_locn = file_name.find('epoch')
            epoch_num  = file_name[epoch_locn:-3]
            title      = 'Embeddings Epoch '+str(epoch_num)
            figures[row,col].set_title(title, size=size_title)

            # set axis label
            if col == 0:
                figures[row,col].set_ylabel('epochs', size=size_axislabel)
            if row == rows-1:
                figures[row,col].set_xlabel('loss', size=size_axislabel)

            # plot it
            figures[row,col].plot(tng_history)

            #savefilename
            savefilename = 'Plot_'+plot_type + '_'+ data_name_5char + '_model_'+ str(save_models.index(file_name)) +'.jpg'

        if not many_plots_on_one_canvas:
            # save the canvas, we'll create a new canvas for the next figure
            suptitle     = plot_type + ' '+ data_name_5char
            finish_plot(canvas, suptitle, savefilename)
        else:
            # get next row and column
            col += 1
            if col > cols-1:
                col = 0
                row += 1           

    if many_plots_on_one_canvas:
        savefilename = 'Plot_'+plot_type + '_'+ data_name_5char +'_AllPlots.jpg'
        suptitle     = plot_type + ' '+ data_name_5char
        finish_plot(canvas, suptitle, savefilename)


#%%
############ INSPECT EMBEDDINGS AS THEY DEVELOP OVER EPOCHS #####################

matrix_of_plots(data_name_5char= '10col',       # '10col' or 'catego'
                test_data_df   = testi_data,    # test data in pandas
                plot_type      = 'embeddings',  # 'embeddings' or 'histories'
                subfolders     = ['SaveModels_MMDVAE', 'latents_by_epoch'],
                charts_per_col = 6, 
                figsize_x      = 10,
                figsize_y      = 10,
                dpi            = 100,
                many_plots_on_one_canvas = False)

#%%

# Now we want to see how 'accounts report' like our reconstructions are.
# We would like points randomly taken from our latent space to be decoded into convincing accounts reports
# However, we note that real data is not normally distributed within the latent space
# So if we rnaomdly defined a matrix through the x and y of the latent space then many points
# would fall well outside the usually distirbution. These are unlikely to decode into convincing accounts reports

# Instead, we can randomly select two points of real data into the latent space
# Then calculate the euclidean path between those two points in 2D space (the line!)
# and calculate points in latent space which are steps along that path.
# Those are our random samples.
# We can shuffle them with the steps between other randomly selected real points
# This gives us a table of random latent points which are within the distribution of
# latent points and reflect how we intend to use the model



#%%
############ FUNCTION TO INSPECT STEPS BETWEEN EXAMPLE COMPANY REPORTS #############################

# Get location of two companies in latent space and then sample steps between those two points
# These steps could be considered the 'business plan' to change from one business to another

def get_steps_btwn(encoder, report_a_npy, report_b_npy, steps=10):
    '''
        Gives steps (ie business plan) between company a and company b
        TAKES: two company reports
        RETURNS: 'step' qty of business reports 
    '''
    # get embeddings of two company reports. encoder outputs latent space x, y.
    # note .numpy() at end of line, as this assumes we get a Tensor back, as is TF2.0's way
    embedding_a    = encoder(report_a_npy).numpy()
    embedding_b    = encoder(report_b_npy).numpy()

    # create dataframe to hold all latent points, kick off with sample a
    z_samples = pd.DataFrame(columns=['latent_x','latent_y'], dtype='float32')

    # To move from a to b we subtract the point a from point b, hence the vector between them
    btwn_vector = embedding_b - embedding_a

    # calculate vector for one step
    btwn_vector_step = btwn_vector / (steps+1)

    # sample 'step' qty of latent points along the vector between a and b
    for step in range(1, steps+1):

        # get next sample and save for future reference
        z_sample  = embedding_a + btwn_vector_step * step
        z_samples = z_samples.append(
                        pd.DataFrame(data   = z_sample, 
                                        columns= ['latent_x','latent_y'], 
                                        index  = pd.Index(['step_%03d'%step], name = 'Report_Ref')
                                    ))

    return z_samples

#%%
### Randomly select reports form test set and get latent points between them

# We assume we must load the models from file...but can't use keras.load_model cos doesn't work on custom classes

## LOAD ENCODER
# instantiate model from scratch
encoder_test     = encoder_mdl(name='test_encoder', params_dict=params_dict)
# compile by passing in data
encoder_test_out = encoder_test(np.array(testi_data[0:5]))
# load weights from file
encoder_test_wts = os.path.join(proj_root, 'SaveModels_MMDVAE', 'data_10col_lossfn_2types_alpha_0.2_norm_False_batch_100_enc_weights.h5')
# apply weights
encoder_test.load_weights(encoder_test_wts)

## LOAD DECODER
# instantiate model from scratch
decoder_test     = decoder_mdl(name='test_decoder', params_dict=params_dict)
# compile by passing in data
decoder_test_out = decoder_test(encoder_test_out)
# load weights from file
decoder_test_wts = os.path.join(proj_root, 'SaveModels_MMDVAE', 'data_10col_lossfn_2types_alpha_0.2_norm_False_batch_100_dec_weights.h5')
# apply weights
decoder_test.load_weights(decoder_test_wts)

## Now continue with getting random points in latent space

# set number of sample reports (from and to). 
n = 5000

# 'n' must be even number less than length of test set
assert (n%2 == 0), "ERROR, n is not even"
assert (len(testi_data) >= n), "ERROR, n is greater than the dataset"

# get sample from test data set
sample_reports = testi_data.sample(n=n, replace=False)

# first half of sample are the 'from' points
from_ref = sample_reports[0:int(n/2)]

# second half of sample are the 'to' points
to_ref   = sample_reports[int(n/2):]

# loop thru random pairs in order (already randomised) and get points between
# each pair of reports (from and to) return 2 latent points between them.
# so we get as many points out as we put in, makes it simple to use
latents_between = [get_steps_btwn(encoder      = encoder_test,
                                  report_a_npy = from_report.reshape(1,22),
                                  report_b_npy = to_report.reshape(1,22),
                                  steps        = 2)
                   for from_report, to_report
                   in zip(np.array(from_ref), np.array(to_ref)) ]

# the above is a list of dataframes, 
# so use concat to bring them together into one dataframe
latents_between = pd.concat(latents_between)

# These are the latents which we can use to decode into reconstructions
# we want to see how 'report like' those reconstructions are vs reconstructions of random data
# We use an autoencoder's reconstruction loss to achieve this comparison
# First let's get those reconstructions....
recons_between = decoder_test(np.array(latents_between)).numpy()

#its convenient if we convert to pandas
recons_between_pd = pd.DataFrame(data=recons_between, columns=testi_data.columns)

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
# We expect error to be approx 1.0 for random data
# A straight autoencoder can get reconstruction error of approx 0.35 with 2 latent dims
# Random data has reconstruction error of approx 1.0

# So, using an Autoencoder trained on the same data, 
# how 'report like' are the reports created by the MMDVAE?

# First let's get a matrix of locations in the latent space and decode into reports

############ FUNCTIONS TO MAKE RANDOM REPORTS #####################
from scipy.stats import truncnorm

def get_truncated_normal(qty, mean=0, sd=1, minimum=-3, maximum=3):

    #get truncated normal (ie normal distrib within a set max and min)
    distribution = truncnorm(a     = (minimum - mean) / sd, 
                             b     = (maximum - mean) / sd, 
                             loc   = mean, 
                             scale = sd)

    # get samples from that distribution
    samples = distribution.rvs(qty)

    return samples

def generate_fake_samples_fromRandoms(dataset, qty):

    # get max and min for each column
    maximums = dataset.max()
    minimums = dataset.min()
    means    = dataset.mean()
    sds      = dataset.std()

    # generate gaussian random data (noise) for each column
    # BUT do so WITHIN the range of the actual data (min, max)
    fake_samples_x = [get_truncated_normal(qty, mean=mean, sd=sd, minimum=mini, maximum=maxi) 
                      for mini, maxi, mean, sd 
                      in zip(minimums, maximums, means, sds)]

    fake_samples_x = np.stack(fake_samples_x, axis=1)

    return fake_samples_x

#%%
# Get Reconstruction Errors for real data.
# We get the mean error from 500 samples of real data and 500 sample sof fake data
# then compare the distributions of the two samples
# This will then allow us to score a batch of reports as being either fake or real

def compare_recon_errs_with_fakes(inputs_to_compare, autoencoder, sample_size=100, batch_qty=20):

    recon_errs_comp_mean = []
    recon_errs_fake_mean = []

    # we take batch_qty batches of the sample_size
    for i in range(1,batch_qty):

        # Get Reconstruction Errors for conparison data
        # sample the comparison data
        inputs_to_compare_sample = np.array(inputs_to_compare.sample(n=sample_size))
        # assume sample_size examples summarized, as would be used in GAN training
        # use training set, not test, because MMDVAE will be using training set
        recons_to_compare= autoencoder.predict(inputs_to_compare_sample)
        recon_errs_comp  = get_reconstruction_losses(inputs  = inputs_to_compare_sample,
                                                     outputs = recons_to_compare)
        
        # Get Reconstruction Errors for fake data
        inputs_fake     = generate_fake_samples_fromRandoms(dataset = testi_data, 
                                                            qty     = sample_size)
        outputs_fake    = autoencoder.predict(inputs_fake)
        recon_errs_fake = get_reconstruction_losses(inputs  = inputs_fake,
                                                    outputs = outputs_fake)

        recon_errs_comp_mean.append(np.mean(recon_errs_comp))
        recon_errs_fake_mean.append(np.mean(recon_errs_fake))

    ## Plot mean reconstruction errors as overlaid histograms

    bins = np.linspace(0, 1.5, 50)
    plt.hist(recon_errs_comp_mean, bins, alpha=0.5, label='real')
    plt.hist(recon_errs_fake_mean, bins, alpha=0.5, label='fake')
    plt.legend(loc='upper right')
    plt.title('Mean Error in '+str(sample_size)+' Fake vs Real Reconstructions')
    plt.show()

    # Print overall mean error to screen
    print("Mean error in batch of real data: ", np.mean(np.array(recon_errs_comp_mean)) )
    print("Mean error in batch of fake data: ", np.mean(np.array(recon_errs_fake_mean)) )

#%%

# load up the autoencoder
file_name = 'data_10col_lossfn_bxent_alpha_0.2_norm_False_batch_32_autoenc_model.h5'
autoencoder = load_model(os.path.join(proj_root,'SaveModels_Autoenc',file_name), compile=False)

#%%
## We can first see how the autoencoder reconstruction errors perform on 
# test set data vs fake data
compare_recon_errs_with_fakes(inputs_to_compare = testi_data, 
                              autoencoder       = autoencoder, 
                              sample_size       = 100, 
                              batch_qty         =  50)

#%%
## Next, let's see the reconstruction error on our decoded latent spaces
compare_recon_errs_with_fakes(inputs_to_compare = recons_between_pd, 
                              autoencoder       = autoencoder, 
                              sample_size       = 100, 
                              batch_qty         =  50)

#%%

### SUMMARY OF RESULTS

# Mean reconstruction error in batch of real test data:         0.626
# Mean reconstruction error in batch of decoded latent points:  0.351
# Mean reconstruction error in batch of fake (random) data:     1.011

# The distribution of the decoded latents does not overlap with random fake data, 
# i.e. its more convincing than random data.
# However, it misses much of the complexity (variance) of the real test data
# Therefore, the autoencoder reconstructs it more successfully
# So basically the MMD VAE is generating simplified reports, 
# which may be tolerable for our use case (business planning)
# But this needs more investigation



#%%

# That's All Folks!




