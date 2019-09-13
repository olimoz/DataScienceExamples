
#%%
################### GENERATIVE ADVERSARIAL NETWORK ###################
# The VAE is only one way we can use deep learning to develop and latent space of company reports
# A common alternative is the Generative Adversarial Network (GAN), famously used for generating face images
# There ar eno examples using business data, but it shoyuld be applicable! 
# So, this code inspired by an example GAN for images at:
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-cifar-10-small-object-photographs-from-scratch/

#%%

# The GAN model is a generator stacked onto a discriminator such that :
#  The generator receives random latent points and generates samples 
#  These are fed into the discriminator which classifies them 
#  The error in the classification is used to update the generator model (not the discriminator)

# The discriminator model will be trained separately, learning from new fake and real examples
# Whereas, we mark the discriminator as 'not trainable' when it is part of the GAN 
# else it may overtrain on fake examples

# We want the discriminator to think that the samples output by the generator are real
# Therefore, when the generator is trained as part of the GAN model,
# we will label the generated samples as real (class 1).

# Why? 
# The discriminator will typically classify generated samples as not real (class 0)
# The backprop process applied to the generator will see this as a large error and
# will update the model weights (i.e. only the weights in the generator) 
# in turn making the generator better at generating good fakes

#%%
################### DIFFERENT TYPES OF GAN ###################

# There are many different variations on the GAN architecture. Which to use?
# Let's recap the following objectives:
#   1. generate business reports
#   2. generate business report paths (ie plans) from a fixed starting point or to a fixed ending point

# We will first build a basic GAN, this can generate business reports well enough
# BUT
# as with all GANs it does not converge easily during training (known as Mode Collapse)
# afterall, its hard to get the discrminator and generator to be well matched, 
# such that one does not always get the upper hand over the other, hence the gan ceases to progress.
# So we will look at the Wasserstein GAN
# BUT
# Neither the basic GAN nor the Wasserstein allow us to control what type of report we can generate
# So we will look at the information maximising GAN, aka infoGAN

#%% 
#########################################################
# NOTES ON WASSERSTEIN GAN
#########################################################

# Our GAN is not improving and simply oscillates
# Whereas the Wasserstein GAN offers convergence in training
# The Wasserstein GAN is best explained at:
# https://machinelearningmastery.com/how-to-implement-wasserstein-loss-for-generative-adversarial-networks/
# and at: https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html#loss-function-for-discriminator

# CRITIC
# In summary the discriminator outputs a smooth score, not a binary 0 or 1
# As such, it is called a 'critic', not a discriminator
# Same architecture as discriminator but has linear activation, not sigmoid

    # fakeness = Dense(1, activation='linear', name='fakeness')(features)
    # def discriminator_loss(y_true, y_pred): return K.mean(y_true * y_pred)
    # discriminator.compile(optimizer=RMSprop(lr=0.00005),loss=[discriminator_loss, 'mse'])

# WASSERSTEIN LOSS
# Furthermore, it applies the 'Earth Mover' distance for the loss. 'Wasserstein' simply refers to this type of distance 
# this is the distance (aka transport) required to move one group of masses from location to another
# hence Earth Mover. It it analogous to the 'transport' required to change one histogram (prb distribution) into another
# We seek the lowest 'transport' to transfer a fake distribution into a real distribution.
# More on this at: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
# The loss must promote ever larger differences between scores for real and generated images.

    # def wasserstein_loss(y_true, y_pred): return K.mean(y_true * y_pred)

# WEIGHT CLIPPING
# The wasserstein loss requires that we constrain critic model weights to a limited range
# after each mini batch update, e.g. [-constant, constant], where constant is a hyperparam, 0.01 to 0.1.

    # for layer in discriminator.layers:
    #   weights = layer.get_weights()
    #   weights = [np.clip(weight, -0.01, 0.01) for weight in weights]
    #   layer.set_weights(weights)

# FREQUENT CRITIC UPDATES
# Update the critic model more times than the generator each iteration (e.g. 5).

# RMSPROP (No momentum)
# Use the RMSProp version of gradient descent with small learning rate and no momentum (e.g. 0.00005).
# It is possible to use momentum, but we need to adapt our model to use 

#%%
#########################################################
# NOTES ON INFO GAN
#########################################################
# We can gain control of the records created by a GAN in two ways:

# Conditional Generative Adversarial Network (CGAN)

    # The generation process can be conditioned, such as via a class label, 
    # so that records of a specific type can be created on demand. 

# Information Maximising GAN (InfoGAN)

    # Provide control variables as input to the generator, along with a point in latent space (noise). 
    # The generator can be trained to use the control variables to influence specific properties 
    # of the generated images. 
    # Those 'control variables' can be classes or vary smoothly. Either way, this is an UNsupervised approach.
    # The control variables are learnt by the network, not imposed.
    # The network is incentivised to maximise the 'Mutual information', 
    # ie the information learned about the control variables given the record generated from the latent space + control variable. 
    # The task of maximizing mutual information is essentially equivalent to training an autoencoder to minimize reconstruction error.

    # Training the generator via mutual information is achieved through the 'auxiliary model'.
    # The auxiliary is simply an appendage to the discriminator
    # IT shares all of the same weights as the discriminator model 
    # but predicts the control codes that were used to generate the image.

    # The function to maximise (not minimise) is:
    # Mutual Info = Entropy(controls) – Entropy(controls | Generator(latents,controls))
    # where:
        # Entropy is a measure of the raw information in a signal. Derived by Shannon
        # Hi entropy = pure, uncompressible, information (eg randomness)
        # Lo entropy = a signal with many patterns, which expand on a small core of information. Easily compressed.
        # Helpfully, The entropy of the control variable is a constant near 0
        # It follows that maximising MI boils down to minimising categorical cross entropy

    # When training the GAN, the controls are randomly selected from a uniform distribution
    # The latents are also randomly selected, but from a normal distribution

    # See https://machinelearningmastery.com/how-to-develop-an-information-maximizing-generative-adversarial-network-infogan-in-keras/


#%%
###################################
# IMPORTS...

import os
import pandas as pd
import numpy as np
import re
from os import listdir, chdir
from os.path import isfile, isdir, join
import time
import datetime
from scipy.stats import norm
import tensorflow as tf
from keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization, concatenate, Dropout
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adam, RMSprop
from keras import metrics
from keras import backend as K
from keras.utils import plot_model, to_categorical
from keras.constraints import Constraint

#%% 
###################################
# GET DATA
# Already prepared in previous Jupyter notebook on Variational Autoencoder

proj_root = 'E:\Data\CompaniesHouse'

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
# Confirm we are running on GPU, not CPU
from tensorflow.python.client import device_lib
print([(item.name, item.physical_device_desc) for item in device_lib.list_local_devices()])

#%%
################### FUNCTION TO GET *REAL* DISCRIMINATOR TRAINING DATA ###################

def generate_real_samples(dataset, qty, smooth_positives=False, overlay_noise=True, GAN_type='standard'):

    # Get randomly selected real data from the encoder's training set
    # shuffle the data
    real_samples   = dataset.sample(n=qty)
    real_samples_x = np.array(real_samples)

    # Get y labels for real data. 1 is real for most models, but -1 is real for Wasserstein!
    if GAN_type == 'Wasserstein':
        real_samples_y = -np.ones((qty, 1))
    else:
        real_samples_y = np.ones((qty,1))
    
    #GAN hack - noisy labels
    if overlay_noise:
        real_samples_y = noisy_labels(real_samples_y, 0.05)

    #GAN hack - smooth positives
    if smooth_positives:
        real_samples_y = smooth_positive_labels(real_samples_y)

    # return
    return real_samples_x, real_samples_y

# using 'label smoothing' on +ve labels. So 0.7 to 1.2 as opposed to 1 vs 0, 
# This is a GAN training hack, see: https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
# This hack * may be * applied when training the discriminator
# BUT we tend not to use this hack because it makes our results worse!

def smooth_positive_labels(y):
    #return y randomly between 0.7 and 1.2 (not 1)
    y_smoothed = y - 0.3 + (np.random.random(y.shape) * 0.5)
    return y_smoothed

# Another GAN training hack is to randomly flip some proportion of labels (default 5%)
# This hack is applied when training the discriminator with fakes and reals.
# This hack greatly improves results with our data, so this is used

def noisy_labels(y, p_flip=0.05):

	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0])

	# choose labels to flip
	flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)

	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]

	return y

#%%
################### FUNCTION TO CREATE RANDOM LATENT VECTORS ###################
# generate random points in latent space as input for the generator

def generate_latent_points(latent_dim, qty, GAN_type='standard', n_cat=None):

    # generate points in the latent space
    z_latent = np.random.normal(size=latent_dim * qty, loc=0, scale=1) # loc=0, scale=1 gives us the standard normal distribution

    # reshape into a batch of inputs for the network
    z_input = z_latent.reshape(qty, latent_dim)

    if GAN_type =='infoGAN':
        # infoGAN's take as input a concatenation of the trng data AND some categories

        # generate categorical codes
        cat_codes = np.random.randint(0, n_cat, qty)

        # one hot encode
        cat_codes = to_categorical(cat_codes, num_classes=n_cat)

        # concatenate latent points and control codes
        z_input = np.hstack((z_input, cat_codes))

        #return both the random latents and the random cat_codes, which will be the y labels.
        return [z_input, cat_codes]
    else:
        # other GAN types need only the latents
        return z_input

#%%
################### FUNCTION TO CREATE *FAKE* TRAINING DATA FOR PRETRAINING DISCRIM ###################

# we pre-train the DISCRIMINATOR with randomly created 'fake' samples
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

# This function is used for creating data with which we can pre-train the discriminator
# it assumes no generator yet exists, there cannot be used to create fake data
# instead, it uses random normal values

def generate_fake_samples_fromRandoms(dataset, qty, overlay_noise=True, GAN_type='standard', n_cat=None):

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

    # infoGAN's take as input a concatenation of the trng data AND some categories
    if GAN_type =='infoGAN':

        # generate categorical codes
        fake_cat_codes = np.random.randint(0, n_cat, qty)

        # one hot encode
        fake_cat_codes = to_categorical(cat_codes, num_classes=n_cat)

        # concatenate latent points and control codes
        fake_samples_x = np.hstack((fake_samples_x, fake_cat_codes))

    # get fake y labels. Zero indicates fake for most GANs, but 1 indicates fake in the Wasserstein
    if GAN_type == 'Wasserstein':
        fake_samples_y = np.ones((qty, 1))
    else:
        fake_samples_y = np.zeros((qty, 1))

    #GAN hack - noisy labels
    if overlay_noise:
        fake_samples_y = noisy_labels(fake_samples_y, 0.05)

    return fake_samples_x, fake_samples_y

#%%
################### FUNCTION TO CREATE *FAKE* TRAINING DATA FOR GAN ###################

# This function is used for creating data with which we can train the discriminator 
# during the gan training process. The discriminator's output then informs training of the generator
# It randomly creates a latent vector which the generator then expands into a fake record
# This is the fake record then presented to the discriminator.

def generate_fake_samples_fromLatent(generator, latent_dim, qty, GAN_type='standard', n_cat=None):
    
    if GAN_type == 'infoGAN':
        # generate points in latent space, 
        # infoGAN generators take a concatenation of the latents and categories, which is provided by the below function
        # The function also reports those categories separately, but we don't use that so _
        z_input, _ = generate_latent_points(latent_dim, qty, GAN_type, n_cat)

    else:
        # other types of generator take only the latents as the input
        z_input = generate_latent_points(latent_dim, qty, GAN_type, n_cat)

    # predict outputs
    fake_samples_x = generator.predict(z_input)
    
    # create 'fake' class labels, 
    # Zero indicates fake for most GANs, but 1 indicates fake in the Wasserstein (where -1 is real)
    if GAN_type == 'Wasserstein':
        fake_samples_y = np.ones((qty, 1))
    else:
        fake_samples_y = np.zeros((qty, 1))
    
    return fake_samples_x, fake_samples_y

#%%
################### HELPERS TO CREATE DISCRIMINATOR ###################

def wasserstein_loss(y_true, y_pred): 
    # Wasserstein loss, only used in Wasserstein GAN
    loss = K.mean(y_true * y_pred)
    
    return loss

# clip model weights in Wasserstein
# we could use np.clip, but that is slow. Better to use Keras Contraints class
class ClipConstraint(Constraint):
    
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
    
    # clip model weights to hypercube
    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)
    
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

#%%
################### FUNCTION TO CREATE DISCRIMINATOR ###################

def define_discriminator(dataset,
                         layer_nodes_discrim,
                         apply_dropout       = False,
                         has_nan             = False,
                         apply_batchnorm     = True,# True = best practise for a GAN
                         leaky_alpha         = 0.2, # 0.2 = best practise for a GAN
                         kinit               = 'glorot_normal',
                         GAN_type            = 'standard',
                         n_cat               = None): # no. of categories for infoGAN's auxiliary model
 
    ## Instantiate inputs ##
    # we'll start with the simplest model, 10cols
    original_dim = len(dataset.columns)

    # Input to discriminator (from data to be modelled)
    inputs_discriminator = Input(shape=(original_dim,), name='discriminator_input')

    # for Wasserstein set weights constraint
    if GAN_type == 'Wasserstein':
        wt_const = ClipConstraint(0.01)
    else:
        wt_const = None

    # Hidden Layers
    discrim = inputs_discriminator

    for nodes in layer_nodes_discrim:
        
        # define layer
        discrim = Dense(nodes, kernel_initializer=kinit, name='discrim_nodes_'+str(nodes), kernel_constraint=wt_const)(discrim)
        
        # optionally apply batch norm
        if(apply_batchnorm):
            discrim = BatchNormalization()(discrim)
        
        # define activation
        discrim = LeakyReLU(alpha=leaky_alpha, name='lkyrelu_discrim_nodes_'+str(nodes))(discrim)

    # Some minor regularisation using dropout
    if apply_dropout:
        discrim = Dropout(0.1)(discrim)

    # define output then compile
    if GAN_type == 'Wasserstein':
        
        # Single Node Discrimator, linear activation (ie smooth score output) 
        # aims to be -ve value for false, +ve for true
        outputs_discriminator = Dense(1, activation='linear', name='critic')(discrim)

        # optimiser
        opt = RMSprop(lr=0.00005)

        # loss is custom loss function
        loss_discrim = wasserstein_loss
    
    elif GAN_type == 'infoGAN':
        
        # Single Node Discrimator, sigmoid activation (ie binary output) 
        # aims to be 0 for false, +ve for true
        outputs_discriminator = Dense(1, activation='sigmoid', name='discriminator')(discrim)

        # We also need an auxiliary model for an infoGAN
        # It shares the same input and hidden layers as a discriminator
        # and is effectively just a new set of output layers on the discriminator
        # it outputs a classification, so uses softmax
        auxiliary_hidden = Dense(16, kernel_initializer=kinit, name='auxiliary_dense1')(discrim)
        auxiliary_hidden = BatchNormalization()(auxiliary_hidden)
        auxiliary_hidden = LeakyReLU(alpha=0.1, name='auxiliary_leakyRelu')(auxiliary_hidden)

        # auxiliary model output, these are categorical codes...
        outputs_auxiliary = Dense(n_cat, activation='softmax', name='auxiliary_output')(auxiliary_hidden)

        # define auxiliary model, but don't compile it.
        # We do't compile because we don't train the discriminator alone, only as part of the gan
        # Compiling the gan causes the auxiliary to be compiled
        auxiliary = Model(inputs=inputs_discriminator, outputs=outputs_auxiliary, name='auxiliary')
        
        # optimiser
        opt = Adam(lr=0.0002, beta_1=0.5)

        # discriminator loss (not auxiliary)
        loss_discrim = 'binary_crossentropy'

    else: # must be standard GAN
        # Single Node Discrimator, sigmoid activation (ie binary output) 
        # aims to be 0 for false, +ve for true
        outputs_discriminator = Dense(1, activation='sigmoid', name='discriminator')(discrim)
        
        # optimiser
        opt = Adam(lr=0.0002, beta_1=0.5)

        # loss
        loss_discrim = 'binary_crossentropy'
       
    # having defined layers, create graph
    discriminator = Model(inputs=inputs_discriminator, outputs=outputs_discriminator, name='discriminator')
    
    # Compile discriminator, using loss defined above
    discriminator.compile(optimizer=opt, loss=loss_discrim, metrics=['accuracy'])
    
    # NOTE, the auxiliary has intentionally note been compiled.

    if GAN_type == 'infoGAN':
        return [discriminator, auxiliary]
    else:
        return [discriminator]

#%%
################### FUNCTION TO TRAIN DISCRIMINATOR ###################
# Some GANs like to pre-train the discriminator using random data
# This function will NOT be used during GAN training, only during pre-training of the discriminator

def train_discriminator(discriminator, dataset, n_iter=20, batch_size=128, 
                        GAN_type='standard', n_cat=None, 
                        smooth_positives=False, overlay_noise=False, wass_clip=0.01):
	
    # ensure we can train weights
    discriminator.trainable = True
    for l in discriminator.layers: l.trainable = True

    mini_batch = int(batch_size / 2)

	# manually enumerate number of batches to train discriminator with
    for i in range(n_iter):

        # get randomly selected REAL samples
        X_real, y_real = generate_real_samples( dataset  = dataset,
                                                qty      = mini_batch,
                                                GAN_type = GAN_type, 
                                                overlay_noise    = overlay_noise, 
                                                smooth_positives = smooth_positives)

        # generate FAKE examples
        X_fake, y_fake = generate_fake_samples_fromRandoms(
                                        dataset      = dataset, 
                                        qty          = mini_batch,
                                        n_cat        = n_cat,
                                        GAN_type     = GAN_type,
                                        overlay_noise= overlay_noise)

        # train discriminator (ie update weights) on REAL samples
        _, real_acc = discriminator.train_on_batch(X_real, y_real)

        # train discriminator (ie update weights) on FAKE samples
        _, fake_acc = discriminator.train_on_batch(X_fake, y_fake)

        if GAN_type == 'Wasserstein': 
            # clip weights
            for layer in discriminator.layers:
                weights = layer.get_weights()
                weights = [np.clip(weight, -wass_clip, wass_clip) for weight in weights]
                layer.set_weights(weights)

        # summarize performance
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

    # ensure weights are fixed as we return control to the gan training process
    discriminator.trainable = False
    for l in discriminator.layers: l.trainable = False

#%%
################### FUNCTION TO DEFINE GENERATOR ###################

def define_generator(   dataset,
                        layer_nodes_generat,
                        reconstruction_cols,
                        GAN_type            = 'standard',
                        has_nan             = False,
                        n_cat               = None,
                        latent_space_nodes  = 2,
                        apply_batchnorm     = False,
                        leaky_alpha         = 0.0,
                        kinit               = 'glorot_normal'):

    if GAN_type == 'infoGAN':
        # infoGAN generators take as input both latent space and categories
        generator_input_size = latent_space_nodes + n_cat
    else:
        # other GAN types take only the latent space as input
        generator_input_size = latent_space_nodes

    # Input to Generator (from latent space)
    inputs_generator = Input(shape=(generator_input_size,), name='generator_input')

    # Generator Hidden Layers
    generated = inputs_generator

    for nodes in layer_nodes_generat:
    
        # define layer
        generated = Dense(nodes, kernel_initializer=kinit, name='decode_nodes_' + str(nodes))(generated)
        
        # optionally apply batch norm
        if(apply_batchnorm):
            generated = BatchNormalization()(generated)
        
        # define activation
        generated = LeakyReLU(alpha=leaky_alpha, name='lkyrelu_decode_nodes_' + str(nodes))(generated)

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
            # define layer to reconstitute a value, uses linear activation (no change)
            reconstruction_layers.append( Dense(1, kernel_initializer=kinit, activation='linear', name=nm)(generated) )
        
        else : # it must be binary
            # define layer to reconstitute a binary, uses sigmoid activation
            reconstruction_layers.append( Dense(1, kernel_initializer=kinit, activation='sigmoid', name=nm)(generated) )

    # concatenate into reconstruction
    outputs_generator = concatenate(reconstruction_layers)

    # instantiate generator model
    generator = Model(inputs = inputs_generator, output = outputs_generator, name = 'generator')

    # GENERATOR IS NOT COMPILED HERE. Compiled as part of GAN.

    return generator

#%%
################### FUNCTION TO DEFINE GAN ###################

def define_gan(discriminator, generator, auxiliary=None, GAN_type='standard'):
    
    # Within the context of the GAN the discriminator weights must be fixed
    # Those weights are only updated when there is an instruction to the discriminator
    # model to fit or train_on_batch
    discriminator.trainable = False
    for l in discriminator.layers: l.trainable = False

    # connect the discriminator to the generator's output
    output_discrim = discriminator(generator.output)

    # define gan in keras functional api
    if GAN_type == 'infoGAN':
        # info gans have two outputs
        # first is the discriminator's opinion of the generator's work (see above)
        # second is the auxiliary's opinion of the generator's work
        output_aux = auxiliary(generator.output)

        # so the info gan will have two outputs, the discrimninator and the auxiliary
        output_gan = [output_discrim, output_aux]

        # for two outputs we need two loss functions
        loss = ['binary_crossentropy', 'categorical_crossentropy']

        # use Adam optimiser for standard gan
        opt = Adam(lr=0.0002, beta_1=0.5)

    elif GAN_type == 'Wasserstein':
        # WGAN's output is the discriminator's output
        output_gan = output_discrim
        
        # WGAN uses wasserstein loss, just like it's discriminator (aka critic)
        loss= wasserstein_loss
        
        # compile with RMS Prop (NO momentum) for Wasserstein
        opt = RMSprop(lr=0.00005)

    else:
        # standard gan has just one output
        # this is the discriminator's opinion of the generator's work
        output_gan = output_discrim

        # use binary cross entropy loss
        loss='binary_crossentropy'

        # use Adam optimiser for standard gan
        opt = Adam(lr=0.0002, beta_1=0.5)

    # define the gan model
    gan = Model(input=generator.input, output=output_gan, name='gan')

    # compile the gan model
    gan.compile(loss=loss, optimizer=opt)

    return gan

#%%
################### FUNCTION TO TRAIN GAN ###################
# Each epoch we get more random points, then train the generator on them
# the discriminator remains fixed (not_trainable)

def train_gan(generator, discriminator, gan, dataset, latent_dim, GAN_type, scoring_model,
              overlay_noise=False, smooth_positives=False, epochs=200, batch_size=64, 
              evaluate_epochs=1, n_cat=None, wass_clip = 0.01, wass_ratio = 5):

    # we train gans batch by batch, not by the more traditional approach of epochs,
    # where each epoch has 'x' number of batches to process in order to pass through all the data.
    # Nevertheless, the concept of an epoch is still useful to quantify how many batches should be used
    batches_per_epoch= int(dataset.shape[0] // batch_size)
    batches_in_trng  = int(epochs * batches_per_epoch)
    half_batch       = int(batch_size//2)

    # For each batch we train the generator we can choose to train the discriminator once or more... 
    if GAN_type == 'Wasserstein':
        # In the Wasserstein the discriminator (aka critic) is trained more often than the generator
        discrim_trng_per_gan_trng = wass_ratio
    else:
        # Other gans train their discrminator only once for each batch on which they train the generator
        discrim_trng_per_gan_trng = 1

    # Now we cycle thru the batches
    # first training the discriminator
    # then training the generator, whilst the discriminator is fixed
    for batch_no in range(batches_in_trng):

        #####################################################################
        # STEP 1: Train discriminator
        #####################################################################
        # For each batch of generator training, decide how many times to train the discriminator
        discrim_iterations = discrim_trng_per_gan_trng

        # The Wasserstein gan occassionally engages in major retraining (100 batches) of the discriminator
        if GAN_type == 'Wasserstein':
            if (batch_no % 1000) < 25 or batch_no % 500 == 0: # 25 times in 1000, every 500th
                discrim_iterations = 5
         
        # ensure we can train discriminator weights
        discriminator.trainable = True
        for l in discriminator.layers: l.trainable = True

        # For each batch we repeat the discriminator training for discrim_iterations
        # which is just 1 for most gans, but more for the Wasserstein
        for discrim_iter in range(discrim_iterations):

            # Now we get samples for the discriminator. Half will be real and half will be fake
            # Get some REAL samples
            X_real, y_real = generate_real_samples(dataset       = dataset, 
                                                   qty           = half_batch, 
                                                   GAN_type      = GAN_type,
                                                   overlay_noise = overlay_noise,
                                                   smooth_positives=smooth_positives)

            # Get some FAKE examples (from latent space)
            X_fake, y_fake = generate_fake_samples_fromLatent(
                                generator  = generator, 
                                latent_dim = latent_dim, 
                                qty        = half_batch, 
                                GAN_type   = GAN_type,
                                n_cat      = n_cat)

            # Update discriminator weights with REAL samples
            d_loss1, _ = discriminator.train_on_batch(X_real, y_real)

            # Update discriminator weights with FAKE samples
            d_loss2, _ = discriminator.train_on_batch(X_fake, y_fake)

            if GAN_type == 'Wasserstein':
                # clip weights
                for layer in discriminator.layers:
                    weights = layer.get_weights()
                    weights = [np.clip(weight, -wass_clip, wass_clip) for weight in weights]
                    layer.set_weights(weights)

        # ensure we cannot train discriminator weights any further
        discriminator.trainable = False
        for l in discriminator.layers: l.trainable = False

        #####################################################################
        # STEP 2: Train Generator in GAN
        #####################################################################
        # Objective is to generate records that convince the discriminator to mark them lowly
        # -1 = perfectly real, but with gradients all the way up to +1 (perfectly fake)
        # In other words, to MINIMISE discriminator output while supplying it with fakes

        # In all GAN types we tell the discriminator that the generated random points are all REAL!
        # For most GAN's real is indicated by y=1 and fake is indicated by 0
        # EXCEPT for the Wasserstein, where real is indicated by -1 and fake is indicated by +1
        if GAN_type == 'Wasserstein':
            y_gan = -np.ones((batch_size, 1))
        else:
            y_gan = np.ones((batch_size, 1))

        # now let's generate the X data and then train the GAN
        if GAN_type == 'infoGAN':
            # infoGAN's have both the data and the categories to prepare, two inputs and two outputs

            # get random points in latent space as input for the generator
            X_gan, cat_codes = generate_latent_points(latent_dim, batch_size, GAN_type, n_cat)

            # train the gan on the batch
            g_loss = gan.train_on_batch(X_gan, [y_gan, cat_codes])

        else:
            # other GAN types have only one output and hence one input
            
            # get random points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, batch_size, GAN_type, n_cat)
            
            # train the gan on the batch
            g_loss = gan.train_on_batch(X_gan, y_gan)

        # evaluate the model performance every evaluate_epochs
        if batch_no % (batches_per_epoch * evaluate_epochs)== 0:
            epoch = batch_no // batches_per_epoch
            summarize_performance(epoch, generator, discriminator, dataset, 
                                  latent_dim, scoring_model, GAN_type, n_cat)

#%%
################### FUNCTION 1 TO MONITOR GAN PERFORMANCE ###################
# When monitoring GAN performane we will use an autoencoder which,
# using reconstruction losses, can quantify how 'fake' the records appear to be
# This function calculates those reconstruction losses.

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
################### FUNCTION 2 TO MONITOR GAN PERFORMANCE ###################
# This function reports on how well the GAN is training

def summarize_performance(epoch, generator, discriminator, dataset, 
                          latent_dim, scoring_model, 
                          GAN_type='standard', n_cat=None, qty=150):
    
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset         = dataset, 
                                           qty             = qty, 
                                           GAN_type        = GAN_type,
                                           overlay_noise   = False,
                                           smooth_positives= False)

    # prepare fake examples
    x_fake, y_fake = generate_fake_samples_fromLatent(generator, latent_dim, qty, GAN_type, n_cat)

    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(X_real, y_real, verbose=0)
    
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

    # The Wassertein discriminator outputs a 'critic' score, not a near binary value
    # So we evaluate it differently

    # evaluate generator using autoencoder trained to identify fake from real
    # Here we submit generated reports to an autoencoder trained only on real reports
    # from the training set. We can then score the generator on the reconstruction errors
    # eperienced by the autoencoder. The higher the error the less familiar the autoencoder was 
    # with the patterns in the data of the report. Meaning, it doesn't resemble a real report.
    # A new autoencoder has been built for this task in another jupyter notebook
    inputs  = x_fake

    # autoencoder has two outputs, regression and binary. Must concatenate predictions
    outputs = np.hstack(scoring_model.predict(x_fake))
    scored_error = np.mean(get_reconstruction_losses(inputs=inputs, outputs=outputs)) - 0.35 #0.35 is the avg autoencoder error for a real record. 

    # summarize performance of discriminator and generator
    print('>Discriminator accuracy: Real=%.0f%%, Fake=%.0f%%. Generator Err: %.0f%%' % (acc_real*100, acc_fake*100, scored_error*100))

    # save the generator model tile file
    filename = 'generator_model_%03d.h5' % (epoch+1)
    generator.save(os.path.join(proj_root, 'SaveModels_GAN', GAN_type, filename))
    
    # if we were using images, then we'd save them now to watch it build ever better samples

#%%
################## FUNCTION TO IDENTIFY COLUMN TYPE ###########################
# The generator demands we know which columns are values and which are binary categories
# We'll use column names to deduce this, but without special characters which are not permitted in keras

def get_col_spec(df):
    '''
        TAKES: the input dataframe
        RETURNS: a dataframe listing column names (without special characters, useful in Keras) and its type
    '''
    colspec = ['binary' if colname.endswith('_ispos') or colname.endswith('_notna') else 'value' for colname in df.columns]
    colname = [re.sub(r'[^\w]', '', colname) for colname in df.columns]
    colspec_df = pd.DataFrame({'col_name' : colname, 'col_type': colspec})
    return colspec_df

#%%
################### FUNCTION TO DEFINE AND TRAIN GAN ###################

def define_and_train_gan(   dataset,              # pandas. Training data. No validation data req'd
                            GAN_type,             # string. 'standard', 'Wasserstein' or 'infoGAN'
                            pretrain_discrim,     # boolean. Whether to pretrain the discriminator or not
                            pretrain_discrim_iters= 20, # no. of batches which which to pre-train the discriminator
                            n_cat                 = None, # int. number of categories for infoGAN 
                            wass_clip             = 0.01, # weight clip for wasserstein
                            wass_ratio         = 5, # how many times the discriminator is trained for each generator training in infoGAN
                            batch_size            = 32, 
                            epochs                = 200,
                            has_nan               = False, # whether the data has binaries indicating nan
                            leaky_alpha_discrim   = 0.2,
                            leaky_alpha_generat   = 0.0,
                            layer_nodes_discrim   = [16, 8, 4, 2],
                            layer_nodes_generat   = [ 2, 4, 8,16],
                            apply_batnorm_discrim = False,
                            apply_batnorm_generat = True,
                            apply_dropout_discrim = False,
                            latent_dim            = 2,
                            overlay_noise         = True, # whether noise should be overlaid on disrim fakes
                            smooth_positives      = False,
                            evaluate_epochs       = 1): # how many epochs between reporting performance
   
    # define the discriminator model
    l = define_discriminator(dataset            = dataset, 
                            layer_nodes_discrim = layer_nodes_discrim,
                            apply_dropout       = apply_dropout_discrim,
                            apply_batchnorm     = apply_batnorm_discrim,
                            has_nan             = has_nan,
                            leaky_alpha         = leaky_alpha_discrim,
                            GAN_type            = GAN_type,
                            n_cat               = n_cat)

    # the define_discriminator function returns a list
    # IF the GAN is an 'infoGAN' then there are two members of the list: 
    # a discriminator model (compiled) and an auxiliary model (uncompiled) 
    # otherwise, only one member: a discriminator model (compiled)
    if GAN_type == 'infoGAN':
        d = l[0] # this is the discriminator (compiled)
        q = l[1] # this is the auxiliary model (uncompiled)
    else:
        d = l[0] # just the discriminator (compiled)
        q = None # no auxiliary available

    # pre-train the discriminator if required.
    # Usually used for standard, but not for Wasserstein or infoGAN
    if pretrain_discrim:
        train_discriminator(discriminator       = d, 
                            dataset             = dataset,
                            n_iter              = pretrain_discrim_iters,
                            batch_size          = batch_size, 
                            GAN_type            = GAN_type,
                            n_cat               = n_cat,
                            wass_clip           = wass_clip,
                            smooth_positives    = smooth_positives, 
                            overlay_noise       = overlay_noise)

    # define the generator model
    g = define_generator(   dataset             = dataset,
                            layer_nodes_generat = layer_nodes_generat,
                            reconstruction_cols = get_col_spec(dataset),
                            GAN_type            = GAN_type,
                            n_cat               = n_cat,
                            apply_batchnorm     = apply_batnorm_generat,
                            leaky_alpha         = leaky_alpha_generat)

    # create the gan
    gan = define_gan(       discriminator       = d,
                            generator           = g,
                            auxiliary           = q,
                            GAN_type            = GAN_type)

    # get pretrained autoencoder to score gan generator results
    # THIS PART IS STILL HARD CODED. SHOULD BE PARAM TO THE FUNCTION
    file_name = 'data_10col_lossfn_mixed_alpha_0.2_norm_False_batch_32_autoenc_model.h5'
    from keras.models import load_model
    autoencoder = load_model(os.path.join(proj_root,'SaveModels_Autoenc','TwoLossModel',file_name), compile=False)

    # fit the gan
    train_gan(  discriminator   = d, 
                generator       = g, 
                gan             = gan, 
                dataset         = dataset,
                latent_dim      = latent_dim, 
                GAN_type        = GAN_type,
                n_cat           = n_cat, 
                wass_clip       = wass_clip,
                overlay_noise   = overlay_noise, 
                smooth_positives= smooth_positives,
                scoring_model   = autoencoder, 
                epochs          = epochs,
                batch_size      = batch_size, 
                evaluate_epochs = evaluate_epochs)

    return d, g, gan

#%%
# EXECUTE DEVELOPMENT OF STANDARD GAN
#######################################

d_stnd, g_stnd, gan_stnd = define_and_train_gan(dataset          = x_train_10cols, 
                                                GAN_type         = 'standard', 
                                                pretrain_discrim = True,
                                                epochs           = 200)

# del d_stnd, g_stnd, gan_stnd

# This WORKS! Final batches read like this:
# >Discriminator accuracy: Real=97%, Fake=100%. Generator Err:  4%
# >Discriminator accuracy: Real=97%, Fake= 98%. Generator Err: -9%
# >Discriminator accuracy: Real=97%, Fake= 89%. Generator Err: 14%

#%%
# EXECUTE DEVELOPMENT OF WASSERSTEIN GAN
#########################################

d_wass, g_wass, gan_wass = define_and_train_gan(dataset          = x_train_10cols, 
                                                GAN_type         = 'Wasserstein', 
                                                pretrain_discrim = False,
                                                wass_clip        = 0.01,
                                                wass_ratio       = 5, # number of times discriminator is trained for each ocassion the generator is trained)
                                                epochs           = 200)
# model is VERY sensitive to wass_clip, the point at which the discriminator weights are clipped

# If wass_clip  =     0.01 then discriminator fails to train at all
# If wass_clip  =     0.1  then discriminator fails to train at all
# If wass_clip  =     1    then discriminator learns real samples, fails to train on fake
# If wass_clip  =    10    then discriminator fails to learn at all
# If wass_clip  =   100    then discriminator fails to learn at all
# If wass_clip  =  1000    then discriminator learns fake samples, fails to train on real
# If wass_clip  = 10000    then discriminator learns real samples, fails to train on fake
# If wass_clip  =100000    then discriminator learns fake samples, fails to train on real

# This experiement was unsuccesful. The GAN did not train.

# del d_wass, g_wass, gan_wass


#%%
# EXECUTE DEVELOPMENT OF INFOGAN
#######################################

d_info, g_info, gan_info = define_and_train_gan(dataset          = x_train_10cols, 
                                                GAN_type         = 'infoGAN', 
                                                pretrain_discrim = False, # infoGAN's are pretrained, but see train_gan() for the implementation
                                                n_cat            = 4, # one for each corner of the 2D latent plane!
                                                epochs           = 200) 

# This takes a while to train, approx 500 iterations until the generator error
# as estimated by the autoencoder's reconstruction error, converges on zero
# Convergence only commenced around 100 iterations, prior to that it explores some wild errors!


#%%

# Evaluate
# The generator tells the discriminator all of its generated images are real
# So, we are seeking generators where the discriminator agrees they are real!
# Hence high accuracy on 'real' is crucial
# It should do this whilst not simply saying everything is real, it would score low on fakes if it did
# Hence, high accuracy on 'fake' is also important, although secondary to accurayc on 'real' samples


#%%
################## FUNCTION TO UN LOG AND UN SCALE ###########################
# When we get results we will want to un-process the daata, ie unscale and unlog
# 
def un_log_scale(df, df_sd, df_mn, reinstate_neg, reinstate_nan):
    '''
        UnLogs and UnStandardises (mult by sd, add mean) value data (not categoricals)
        To be used to inspect results from model and compare with other results
        TAKES: dataframe after log and scale has been applied
        RETURNS: dataframe unlogged and unscaled
    '''

    ## Ensure we have the same columns in data as in means and sd's
    # get non categorical data
    data = get_value_cols_only(df)

    # get columns in same order
    data_cols_orig = data.columns.to_list()
    index = data.index
    data_cols = data.columns.sort_values().to_list()
    data      = data[data_cols]
    sd_cols   = df_sd.reset_index().sort_values(by='index')
    sd_cols.columns = ['colname','sd']
    mn_cols   = df_mn.reset_index().sort_values(by='index')
    mn_cols.columns = ['colname','mn']

    # assert columns are same
    assert (data_cols == sd_cols['colname'].to_list()), "ERROR, sd columns do not match"
    assert (data_cols == mn_cols['colname'].to_list()), "ERROR, mean columns do not match"

    # proceed to unscale
    # note, * is elementwise multiplication using broadcasting where necessary, which is what we want
    # np.multiply would be the same whereas np.dot applies matrix multiplication
    data = data.to_numpy() * sd_cols['sd'].to_numpy() + mn_cols['mn'].to_numpy()

    # proceed to unlog
    data = np.exp(data)

    # and subtract 1 (afterall, we originally did log(1+value))
    data = data - 1

    ## return to pandas
    data = pd.DataFrame(data=data, columns=data_cols, index=index)
    data = data[data_cols_orig]

    # select key columns which have been un-scaled and unlogged first
    component1 = data[['MaxClose','Duration']]

    # other columns which have been un-scaled and un-logged
    component2 = data[[colname for colname in data_cols_orig if colname not in ['MaxClose','Duration']]]

    # columns which were not unscaled/logged, ie categoricals
    component3 = df[[colname for colname in df.columns if colname not in data_cols_orig]]

    # we can also reinstate negatives and nans using this function
    def reinstate(component2, component3, col_suffix, to_replace, new_value):

        #save the original column order
        orig_cols = component2.columns

        # get data columns and their respective category columns in same order
        is_cat = component3 >> select(ends_with(col_suffix))

        # sort columns to ensure same order
        is_cat = is_cat[is_cat.columns.sort_values().to_list()]
        component2 = component2[component2.columns.sort_values().to_list()]

        # assert columns are correct for multiplication
        assert ([col.replace(col_suffix, '') for col in is_cat.columns.to_list()] == component2.columns.to_list()), "ERROR, ispos columns do not match"

        # apply replacements as required
        is_cat = is_cat.replace(to_replace=to_replace, value=new_value)

        # apply the multiplication in numpy
        component2_data = component2.to_numpy() * is_cat.to_numpy()

        # revert to pandas
        component2 = pd.DataFrame(data=component2_data, columns=component2.columns)

        # apply original column order
        component2 = component2[orig_cols]

        return component2

    # optionally reinstate -ve's
    if reinstate_neg:
        # reinstate negatives
        component2 = reinstate(component2, component3, col_suffix='_ispos', to_replace=0, new_value=-1)
        
        # we opted to reinstate -ves so we should  superfluous categorical columns
        cols_to_ = [colname for colname in component3.columns if colname.endswith('_ispos')]
        component3 = component3.drop(columns=cols_to_)

    # optionally reinstate Nans
    if reinstate_nan:
        # reinstate Nans
        component2 = reinstate(component2, component3, col_suffix='_notna', to_replace=0, new_value=np.nan)
        
        # we opted to reinstate Nans so we should  superfluous categorical columns
        cols_to_ = [colname for colname in component3.columns if colname.endswith('_notna')]
        component3 = component3.drop(columns=cols_to_)

    # recombine with categoricals and return to pandas
    # Pay special attention to reset_index(=True) for each component
    # If this is not done then the indexes force unusual behaviour and an irregular table
    # this section may be better done using dfply (sql style) joins rather than pd.concat
    data_return = pd.concat([component1.reset_index(drop=True), 
                             component2.reset_index(drop=True),
                             component3.reset_index(drop=True)], 
                             axis=1)
    data_return.index = index
    
    return data_return

#%%
############ FUNCTION TO INSPECT GRID OF RANGE OF COMPANY REPORTS ##################################

# Our first example of being 'generative'
# build a report generator that samples a grid of generated companies
# by decoding points from latent space
# Those points are selected to represent a normal distribution in the x and y latent spaces

def get_grid_of_samples(df, df_sd, df_mn, generator, reinstate_neg, reinstate_nan):
    '''
        Builds matrix of latent values and generates company reprot for each one
        TAKES: a df of test data (for its columns), data means, data sd's, a generator
        RETURNS: a dataframe of reports
    '''
    # create array of zeros as blank company report
    granularity = 10

    # get equally spaced range of probabilities, from 5% to 95%
    prob_rng = np.linspace(0.05, 0.95, 10)

    # get the x axis value of the normal distribution with that probability
    # http://eric.univ-lyon2.fr/~ricco/tanagra/fichiers/en_Tanagra_Calcul_P_Value.pdf
    gen_latent_x = norm.ppf(prob_rng)
    gen_latent_y = norm.ppf(prob_rng)

    # create blank table for results
    colnames  = df.columns
    recon_rpts= df[0:0]
    z_samples = pd.DataFrame(columns=['latent_x','latent_y'], dtype=float)

    # iterate around the grid getting samples at each point
    for x in gen_latent_x:
        for y in gen_latent_y:

            # get next sample and save for future reference
            z_sample = np.array([[x, y]])
            z_samples= z_samples.append(pd.DataFrame(data=z_sample, columns=['latent_x','latent_y']))
            
            # reconstruct company report from latent space representation
            recon_rpt = generator.predict(z_sample)

            # save company report for bulk unscale / unlog later
            recon_rpt = pd.DataFrame(data=recon_rpt, columns=colnames)
            recon_rpts= recon_rpts.append(recon_rpt)

    # unlog and unscale the data
    recon_rpts_unlog = un_log_scale(df    = recon_rpts,
                                    df_sd = df_sd,
                                    df_mn = df_mn,
                                    reinstate_neg = reinstate_neg,
                                    reinstate_nan = reinstate_nan)

    # return the embeddings (z_smaples) along side their unscale/unlogged company report
    recon_rpts_unlog = pd.concat([z_samples, recon_rpts_unlog] , axis=1)

    return recon_rpts_unlog

#%%
############ INSPECT GRID OF RANGE OF COMPANY REPORTS ##################################
x_testi_grid_gan = get_grid_of_samples( df        = x_testi,
                                        df_sd     = sds_10cols,
                                        df_mn     = means_10cols,
                                        generator = generator,
                                        reinstate_neg = True,
                                        reinstate_nan = False)

x_testi_grid_gan.to_csv(os.path.join(proj_root, 'SaveModels_GAN', x_testi_10cols_grid_gan.csv'), mode='w', header=True, index=False)


#%%
# Get the initial discriminator training data
discrim_x, discrim_y = get_discrim_tng_data(  
                                            df      = x_train,
                                            df_sd   = sds, 
                                            df_mn   = mns, 
                                            generator = enc, 
                                            qty     = 10000,
                                            reinstate_neg = True, 
                                            reinstate_nan = False
                                            )
discrim_x['y'] = discrim_y                      

# split into training and validation
discrim_trn, discrim_val = train_test_split_df(discrim_x, propns=[0.8, 0.2], seed=20190703)

discrim_trn_x = discrim_trn.drop(columns=['y'])
discrim_val_x = discrim_val.drop(columns=['y'])

discrim_trn_y = discrim_trn['y']
discrim_val_y = discrim_val['y']

# instantiate the discriminator
descriminator = define_and_compile_discriminator(
                                            input_df            = discrim_trn_x,
                                            layer_nodes_discrim = [8, 4, 2],
                                            has_nan             = False,
                                            apply_batchnorm     = False,
                                            leaky_alpha         = 0.2, # best practise for a GAN
                                            kinit               = 'glorot_normal')
# train the discriminator
discriminator_history = descriminator.fit(  x          = discrim_trn_x,
                                            y          = discrim_trn_y,
                                            epochs     = 20,
                                            batch_size = 100,
                                            verbose    = 2,
                                            shuffle    = True,
                                            validation_data = (discrim_val_x, discrim_val_y),
                                          )

