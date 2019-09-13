

### INTRODUCTION

# The VQ-VAE is a type of variational autoencoder (VAE), 
# but the latent code Z goes through a discrete bottleneck 
# before being passed to the decoder.

# The bottleneck matches...
#     the latent code provided by the encoder
#     to its nearest neighbor in a codebook

#     We update the codes in the codebook each mini-batch
#     The updated values are the exponential moving averages (EMA) 
#     of the recent mini batches latent codes and the matching codebook values

#     In effect, the codebook is the EMA of the previous latent codes.
#     We permit a limited number of these codes.

# We call this 'vector quantization'

# To train, we minimize the weighted sum of 
#     the reconstruction loss 
#    and 
#    a commitment loss that ensures the encoder commits to entries in the codebook. 

# This code was inspired by VQ VAE code which can be found in R at RStudio's blog
# This blog explains the VQ-VAE well, and is much more useful than jumping 
# straight into the Python version, below, which was presente dby the Tensorflow team.
#   https://blogs.rstudio.com/tensorflow/posts/2019-01-24-vq-vae/

# The above R code is a shortened edition of the Python code presented by the Tensorflow team at:
#   https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vq_vae.py

# The OOP approach is introduced at:
#   # https://www.tensorflow.org/beta/guide/keras/custom_layers_and_models#building_models

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#%%
#### IMPORTS

import pandas as pd
import numpy as np
from os import listdir, chdir
from os.path import isfile, isdir, join
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.python.training import moving_averages
from tensorflow.keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization, Concatenate, Reshape
from tensorflow.keras import Model
import cProfile

#%% 
#### Confirm environment

print(tf.__version__)
tf.executing_eagerly()

#%%
### LOAD DATA

proj_root = 'E:\Data\CompaniesHouse'

prepped_10cols = pd.read_csv(join(proj_root, 'prepped_10cols.csv'))
sds_10cols     = pd.read_csv(join(proj_root, 'sds_10cols.csv'), index_col=0)
means_10cols   = pd.read_csv(join(proj_root, 'means_10cols.csv'), index_col=0)

x_train_10cols = pd.read_csv(join(proj_root, 'x_train_10cols.csv'), index_col='Report_Ref')
x_train_10cols = x_train_10cols.drop(columns=['Unnamed: 0'])

x_valid_10cols = pd.read_csv(join(proj_root, 'x_valid_10cols.csv'), index_col='Report_Ref')
x_valid_10cols = x_valid_10cols.drop(columns=['Unnamed: 0'])

x_testi_10cols = pd.read_csv(join(proj_root, 'x_testi_10cols.csv'), index_col='Report_Ref')
x_testi_10cols = x_testi_10cols.drop(columns=['Unnamed: 0'])



#%%
# number of embedding vectors
num_codes = 64

# dimensionality of the embedding vectors
code_size = 4

# The latent space in our example will be of size one, 
# that is, we have a single embedding vector representing 
# the latent code for each input sample
latent_space_dims = 1

# nodes of the layer calculating latent space
latent_space_nodes = latent_space_dims * code_size


#%%
############ ENCODER #################

# Coded in OOP format as a custom tensorflow model, as opposed to static functions
# This coding approach is 'pythonic' and reminiscent of pytorch
# BUT
# its harder to debug

class encoder_mdl(tf.keras.Model):

    def __init__(self, params_dict, name, **kwargs):
        super(encoder_mdl, self).__init__(name=name, **kwargs)

        # Get params
        self.original_dim       = params_dict.get('original_dim')
        self.layer_nodes_list   = params_dict.get('layer_nodes_list') # is list, not single object
        self.latent_space_dims  = params_dict.get('latent_space_dims')
        self.code_size          = params_dict.get('code_size')
        self.kinit              = params_dict.get('kinit')
        self.leaky_alpha        = params_dict.get('leaky_alpha')
        self.apply_batchnorm    = params_dict.get('apply_batchnorm')

        # define possible encoder layers, there can be any number, 
        # eg 3 layers: 16 nodes, 8 nodes, 2 nodes. Or only 2 layers: 32 nodes, 16 nodes.
        # Therefore, we use exec() to define the class variables dynamically
        for layer_nodes in self.layer_nodes_list :
            i = self.layer_nodes_list.index(layer_nodes)
            
            # define dense layer
            exec("self.enc_lyr_{} = Dense(layer_nodes, kernel_initializer=self.kinit, name='enc_lyr_'+str({}))".format(i, i))
            
            # define leaky activation for above dense layer
            exec("self.enc_lyr_lky_{} = LeakyReLU(alpha=self.leaky_alpha, name='enc_lyr_lky'+str({}))".format(i, i))

        # define a batch norm layer
        if self.apply_batchnorm:
            self.enc_batchnorm = BatchNormalization()

        # define final dense layer for input to latent space
        self.latent_dense     = Dense(self.latent_space_dims * self.code_size, name='latent_dense')
        self.latent_dense_lky = LeakyReLU(alpha=self.leaky_alpha, name='latent_dense_lky')
        
        # reshape output to latent space * code_size
        self.latent_reshape   = Reshape((self.latent_space_dims, self.code_size))

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        encoded = inputs

        for layer_nodes in self.layer_nodes_list:
            i = self.layer_nodes_list.index(layer_nodes)
            
            # define dense looped layer, can be a varying numbe rof these, so use exec()
            exec("encoded = self.enc_lyr_{}(encoded)".format(i))
            
            # optionally apply batch norm
            if self.apply_batchnorm:
                exec("encoded = self.enc_batchnorm(encoded)")

            # define leaky activation for above dense layer
            exec("encoded = self.enc_lyr_lky_{}(encoded)".format(i))

        encoded = self.latent_dense(encoded)
        
        encoded = self.latent_dense_lky(encoded)

        encoded = self.latent_reshape(encoded)

        return encoded

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
        shape = [self.latent_space_dims, self.code_size]
        return tf.TensorShape(shape)

#%%
## Let's test it!

# set params
params_dict =  { 
                'original_dim'       : 22,
                'layer_nodes_list'   : [16,8],
                'latent_space_dims'  : latent_space_dims,
                'code_size'          : code_size,
                'kinit'              : 'glorot_normal',
                'leaky_alpha'        : 0.2,
                'apply_batchnorm'    : False,
                } 

# instantiate the new model class
encoder = encoder_mdl(name='my_encoder', params_dict=params_dict)

# execute it with a few lines of training data
# NOTE np.array is fed to inputs, NOT pandas table
test = encoder(np.array(x_train_10cols[0:5]))

# now let's view a summary, see if its as expected
encoder.summary()

#%%



#%%
############ VECTOR QUANTISER #################
# This is not a layer nor a model
# It simpy extends 'object'

class vector_quantiser_model(object):

    def __init__(self, num_codes, code_size):

        self.num_codes = num_codes
        self.code_size = code_size

        # store for the embedding vectors
        # Note, all set as 'untrainable'
        # We do the updates via a moving average function, not via gradient descent
        self.codebook  = tf.Variable(name = 'codebook',
                                     shape = [num_codes, code_size], 
                                     dtype = tf.float32)

        self.ema_count = tf.Variable(name = "ema_count", 
                                     shape = [num_codes],
                                     initializer = tf.constant_initializer(0),
                                     trainable = False)

        self.ema_means = tf.Variable(name = "ema_means",
                                     initializer = self.codebook.initialized_value(),
                                     trainable = False)

    def __call__(self, codes):
        # Match encoder output to available embeddings
        # Returns nearest_codebook_entries, one_hot_assignments

        # codes = the latent vectors to be compared to the codebook.
        # codes have shape [batch_size, latent_size, code_size]

        # First, we compute the Euclidean distance of each encoding to the vectors in the codebook
        # shape: (batch , 1 , num_codes)
        distances <- tf.norm(
                tensor = # encodings
                         tf.expand_dims(tensor=codes, axis=2) - # add a dimension as the 3rd dim (0,1,2)
                         # less vectors in code book
                         tf.reshape(tensor = self.codebook, 
                                    shape  = (1, 1, self.num_codes, self.code_size)),
                axis = 3, # computes norms over fourth dim, code_size
                ord  = 'euclidean')

        # We assign each vector to the closest embedding 
        # shape: (batch, 1)
        assignments = tf.argmin(distances, axis = 2) # axis2=num_codes

        # This gives us the one-hot vectors corresponding to the matched
        # codebook entry for each code in the batch.
        # shape: (batch, 1, num_codes)
        one_hot_assignments = tf.one_hot(indices=assignments, depth=self.num_codes)
        
        # Isolate the corresponding vector by masking out all others 
        # shape: (batch, 1, code_size)
        nearest_codebook_entries = tf.reduce_sum(
                        # get the mask
                        tf.expand_dims(tensor=one_hot_assignments, axis=-1) * 
                        # multiply it by the codebook (shaped appropriately)
                        tf.reshape(tensor=self.codebook, 
                                   shape=(1, 1, self.num_codes, self.code_size)),
                        # sum up the result in order to un-one-hot the vector
                        axis = 2)

        return nearest_codebook_entries, one_hot_assignments

#%%
# Instantiate Vector Quantiser

vector_quantiser = vector_quantiser_model(num_codes = num_codes, code_size = code_size)

#%%

# Method for updating codes. These are not learned via gradient descent
# Instead, they are exponential moving averages, 
# continually updated by whatever new “class member” is assigned to them

# Heavily depends on use of assign_moving_average() from tensorflow.python.training.moving_averages
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/moving_averages.py

# This function computes the moving average of a variable thus:
#   variable = variable - ((1 - decay) * (variable - value))

# function to calculate moving averages of codebook values:
def update_ema(vector_quantiser, one_hot_assignments, codes, decay=0.99):
    
    # EMA = ema of sum / ema of qty

    # Count the quantity of currently assigned samples per codebook entry
    updated_ema_count = moving_averages.assign_moving_average(
                            variable = vector_quantiser.ema_count,
                            value    = tf.reduce_sum(one_hot_assignments, axis = [0, 1]),
                            decay    = decay,
                            name     = 'updated_ema_count',
                            zero_debias = False)

    # Calculate the the new EMA of those codebook entries
    # This selects all assigned values (masking out the others) and sums them up over the batch
    # This will be divided by count later, so we get an average
    updated_ema_sums = moving_averages.assign_moving_average(
                            variable = vector_quantiser.ema_means,
                            value    = tf.reduce_sum(
                                            # assigned values
                                            tf.expand_dims(tensor=codes, axis=2) *
                                            # masking out others
                                            tf.expand_dims(tensor=one_hot_assignments, axis=3), 
                                            axis = [0, 1]),
                            decay    = decay,
                            name     = 'updated_ema_means',
                            zero_debias = FALSE)

    # prevent division by zero
    updated_ema_count = updated_ema_count + 1e-5

    # calculate new means. EMA = ema of sum / ema of qty
    updated_ema_means = updated_ema_sums / tf.expand_dims(updated_ema_count, axis = -1)

    # assign updated values to codebook
    # prior to tf2.0 this read: tf.assign(vector_quantizer.codebook, updated_ema_means)
    # but with eager execution we can reference the variables directly
    vector_quantiser.codebook = updated_ema_means

#%%
################ DECODE ##################
# Note, the decoder does not output logits for eahc field
# instead it outputs probabilities for each field
# This is done using Tensorflow Probability
# See https://blogs.rstudio.com/tensorflow/posts/2019-01-08-getting-started-with-tf-probability/

class decoder_mdl(tf.keras.Model):

    def __init__(self, params_dict, name, **kwargs):
        super(decoder_mdl, self).__init__(name=name, **kwargs)

        # Get params
        self.original_dim       = params_dict.get('original_dim')
        self.layer_nodes_list   = params_dict.get('layer_nodes_list') # is list, not single object
        self.latent_space_dims  = params_dict.get('latent_space_dims')
        self.code_size          = params_dict.get('code_size')
        self.kinit              = params_dict.get('kinit')
        self.leaky_alpha        = params_dict.get('leaky_alpha')
        self.apply_batchnorm    = params_dict.get('apply_batchnorm')

        # reshape output to (1, latent space * code_size)
        self.latent_reshape   = Reshape((1, self.latent_space_dims * self.code_size))

        # define final dense layer for input to latent space
        self.latent_dense     = Dense(self.latent_space_dims * self.code_size, name='latent_dense')
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
            exec("self.dec_lyr_{} = Dense(layer_nodes, kernel_initializer=self.kinit, name='dec_lyr_'+str({}))".format(i, i))
            
            # define leaky activation for above dense layer
            exec("self.dec_lyr_lky_{} = LeakyReLU(alpha=self.leaky_alpha, name='dec_lyr_lky'+str({}))".format(i, i))

        # define reconstruction layer(s)
        # define layer to reconstitute the values (fields 1 to 12)
        self.recon_regres = Dense(12, kernel_initializer=self.kinit, activation='linear',  name='recon_regres')
        
        # define layer to reconstitute a binary, uses sigmoid activation (fields 13 to 22)
        self.recon_binary = Dense(10, kernel_initializer=self.kinit, activation='sigmoid', name='recon_binary')

        # concatenate before presentation as probability
        self.reconstruction = Concatenate()

    def call(self, latent):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        decoded = latent

        decoded = self.latent_reshape(decoded)

        decoded = self.latent_dense(decoded)

        decoded = self.latent_dense_lky(decoded)

        # optionally apply batch norm
        if self.apply_batchnorm:
            exec("decoded = self.dec_batchnorm(decoded)")

        # define possible decoder layers, there can be any number, 
        # eg 3 layers: 4 nodes, 8 nodes, 16 nodes. Or only 2 layers: 5 nodes, 10 nodes.
        # Therefore, we use exec() to define the class variables dynamically
        for layer_nodes in self.layer_nodes_list:
            i = self.layer_nodes_list.index(layer_nodes)
            
            # define dense looped layer, can be a varying numbe rof these, so use exec()
            exec("decoded = self.dec_lyr_{}(decoded)".format(i))
            
            # define leaky activation for above dense layer
            exec("decoded = self.dec_lyr_lky_{}(decoded)".format(i))

        # There are two outputs, values for regression (mse)...
        output_regres = self.recon_regres(decoded)

        # ...and binaries for cross entropy
        output_binary = self.recon_binary(decoded)

        # concatenate so there is only one output object
        logits = self.reconstruction([output_regres, output_binary])

        # Output will not be logits, but rather a probability for each field
        return tfd.Independent( 
                        distribution = tfd.Bernoulli(logits=logits), #if data was pixels then this would be Bernoulli
                        reinterpreted_batch_ndims = len([original_dim]),
                        name="decoder_distribution")

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
        shape = [[12],[10]]
        return tf.TensorShape(shape)

#%%
# Test it!

# set params
params_dict =  { 
                'original_dim'       : 22,
                'layer_nodes_list'   : [8,16],
                'latent_space_dims'  : latent_space_dims,
                'code_size'          : code_size,
                'kinit'              : 'glorot_normal',
                'leaky_alpha'        : 0.2,
                'apply_batchnorm'    : False,
                } 

# instantiate the new model class
decoder = decoder_mdl(name='my_decoder', params_dict=params_dict)

# execute it with a few lines of training data
# NOTE np.array is fed to inputs, NOT pandas table
# test2 = decoder()

# now let's view a summary, see if its as expected
# encoder.summary()

#%%
########## TRAINER ############

num_epochs = 20
batch_size = 32

# loop over epochs
for epoch in range(num_epochs):

    # loop over batches in a single epoch
    for batch in len(x_train_10cols)//batch_size + 1 :
    
        # rules for start of epoch
        if batch == 0:
            batch_start = 0
            batch_end   = batch_size
        else:
            batch_start = batch_end 
            batch_end   = batch_end + batch_size

        # rules for end of epoch
        if batch_start > len(x_train_10cols):
            break

        if batch_end > len(x_train_10cols):
            batch_end = len(x_train_10cols)

        # get batch
        x = x_train_10cols_10cols[batch_start:batch_end,]

        # the training will require us to intervene in the normal process of calculating gradients
        # this is because we will be calculating codebook entries in a funciton which is not part of 
        # normal training using gradients
        # The codebook, ema_count and ema_means are are tf.Variables, in order to update them
        # we need to 'watch' them during the training process
        # The tool for doing this is tf.GradientTape
        with tf.GradientTape(persistent = TRUE) as tape:
        
            #### FORWARD PASS ###
            # encoder the batch
            codes = encoder(x)
            
            # assign codes to codebook and update codebook
            # don't update the codebook, just get the closest vector
            nearest_codebook_entries, one_hot_assignments = vector_quantizer(codes)

            # Training gradients cannot pass through this codebook assignment, its not differentiable
            # So how will we back prop errors from the decoder back into the encoder?
            # We must instruct the forward and back propogation to go around the codebook assignment
            # The tool for that is tf.stop_gradient
            codes_straight_through = codes + tf.stop_gradient(nearest_codebook_entries - codes)
            
            # now get output from the decoder
            # This is a distribution, not a single figure
            decoder_distribution   = decoder(codes_straight_through)

            # To recap, backprop will take care of the weights in the decoder and encoder
            # whereas the codebook (ie latent embeddings) are updated using moving averages

            ### Calculate loss ###
            # 1. Reconstruction Loss
            # The log probability of the actual input for the batch, 'x', vs the distribution learned by the decoder, 'decoder_distribution'
            reconstruction_loss = -tf.reduce_mean(decoder_distribution.log_prob(x))

            # 2. Commitment Loss
            # The mean squared deviation of 
            #   the encoded input samples from 
            #   the nearest neighbors they’ve been assigned to:
            # Note, minimising this loss incentivises the model to produce a concise set of latent codes!
            commitment_loss = tf.reduce_mean(tf.square(codes - tf.stop_gradient(nearest_codebook_entries)))
            
            # Finally, we have the usual KL divergence loss
            # As, a priori, all assignments are equally probable,
            # this component of the loss is constant and can oftentimes be dispensed of. 
            # We’re adding it here mainly for illustrative purposes.
            prior_dist = tfd.Multinomial(total_count = 1,
                                         logits = tf.zeros([latent_size, num_codes]))
            prior_loss <- -tf.reduce_mean(
                                tf.reduce_sum(
                                    prior_dist.log_prob(one_hot_assignments), 
                                1))
            
            # Sum the losses
            beta = 0.25
            loss = reconstruction_loss + beta * commitment_loss + prior_loss
        
        # end of forward prop for the batch, 
        # now calculate gradients for encoder a
        encoder_gradients = tape.gradient(loss, encoder.variables)
        
        # and gradients for decoder
        decoder_gradients = tape.gradient(loss, decoder.variables)
        
        # apply the gradients to the ENCODER weights using the optimiser's learning rate
        grads_and_vars_enc = list(zip(encoder_gradients, encoder.variables))
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_enc,
                                  global_step    = tf.train.get_or_create_global_step())

        # apply the gradients to the DECODER weights using the optimiser's learning rate
        grads_and_vars_dec = list(zip(decoder_gradients, decoder.variables))
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_dec,
                                  global_step    = tf.train.get_or_create_global_step())

        # update the codebook, using exponential moving average
        update_ema(vector_quantiser = vector_quantiser,
                one_hot_assignments = one_hot_assignments,
                codes = codes,
                decay = decay)

        # end of batch loop
        
    # end of epoch loop

#%%

# That's All Folks!




