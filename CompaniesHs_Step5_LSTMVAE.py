#%%
### LSTM AUTOENCODERS, FIRST WITH VAE SECOND WITH MMD

# So far we have considered a latent space of company reports.
# But many companies present us with a time series of company reprots, 
# not just one-off reports.

# We could build a latent space of time series of company reports, 
# thereby capturing the trajectory of company results

# At first it is tempting to consider a latent space with a handful of dimensions 
# for the compressed features but also with a time dimension, undoctored (uncompressed)
# This would allows us to easily view the trajectories in latent space

# BUT

# This is effectively what we already have using a latent space of single reports.
# We simply need to select those points by company and draw the trajectory over time

# The proposal here is different. It is a latent space where one point represents the 
# entire trajectory of company reports. It will necessarily be of a higher number of dimensions
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
#   the MMD (overlapping gaussians as structure)

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

import math       as math
import pandas     as pd
import numpy      as np
import tensorflow as tf

from dfply                   import *
from tensorflow.keras        import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda, LeakyReLU, BatchNormalization, Concatenate, Reshape, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils  import plot_model

#%% 
#### Confirm environment

print(tf.__version__)
tf.executing_eagerly()

#%%
### LOAD DATA

proj_root  = 'E:\\Data\\CompaniesHouse'
subfolders = ['SaveModels_AutoencLSTM_VAE']

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
### DATA WRANGLING ###

# We will have to completely reform the data for a seq2seq model
# so, first compile into one single dataset

all_data = pd.concat([x_train_10cols, x_testi_10cols, x_valid_10cols])

#%% 
# Any duplicates in the Report_Ref ?
test = all_data.reset_index(drop=False) >> \
            select(X.Report_Ref) >> \
                group_by(X.Report_Ref) >> \
                    summarize(Refcount = X.Report_Ref.count())

test >> filter_by(X.Refcount>1) 

#%%
# No duplicates, great, let's continue...

## Reconfigure data into sequences for each company
# the index is company id and date of report, lets get those into columns of their own
all_data                = all_data.reset_index(drop=False)
all_data['crn']         = all_data['Report_Ref'].str[0:8]
all_data['report_date'] = all_data['Report_Ref'].str[9:].astype('int')

# get unique report crn and date combinations
UniqueCRN_Date = all_data >> \
                    select('crn', 'report_date') >> \
                        distinct('crn', 'report_date')

# Get chronological sequence number for each report for a given company; 1,2,3,4 etc
# Step1. Get sequence using pandas equivalent of ROW_NUMBER() OVER(ORDER BY report_date PARTITION BY crn)
# Also get time between reports, may have skipped a year, or be much less than a year
UniqueCRN_Date_Count = (UniqueCRN_Date.assign(
                                    sequence = UniqueCRN_Date.sort_values(['report_date'], ascending=True)
                                        .groupby(['crn'])
                                        .cumcount() + 1)
                                .sort_values(['crn','report_date','sequence'])) >> \
                        mutate(TimeDelta = X.report_date - lag(X.report_date, 1))                       

# Step2. Ensure TimeDelta is sensible where sequence =1
UniqueCRN_Date_Count['TimeDelta']= UniqueCRN_Date_Count.apply(
                                    lambda row : 
                                        10000 if row['sequence']==1 else row['TimeDelta'], 
                                        axis = 1)

# How many reports in a sequence is enough for sequence modelling?
# we'll need at least one for output (forecast)
# So minimum is assuemd to be 4 totol, 3 input and one output.
report_yrs = 4

# Step3. Identify companies where there is a TimeDelta <>12 months (10000)
# within the first 4yrs
CRN_non_sequential = UniqueCRN_Date_Count >> \
                        filter_by(X.TimeDelta != 10000.0, X.sequence <= report_yrs) >> \
                            distinct(X.crn)

# Step4. Remove those company's reports from the list of useful companies
UniqueCRN_Date_Count = UniqueCRN_Date_Count >> \
                        anti_join(CRN_non_sequential, by='crn')

# Step5. Identify companies with >3 sequential reports available
CRN_with_Sufficient_Reports = UniqueCRN_Date_Count >> \
                                filter_by(X.sequence >= report_yrs) >> \
                                    distinct(X.crn)

print("Training data has ",len(CRN_with_Sufficient_Reports),' companies with at least 4 sequential reports')

#%%

# This is NOT ENOUGH data to train a deep learning model!
# But in the expectation of more data from SEC, we'll build the model anyway

#%%
# select the first 4 reports of those companies

# filter down to companies with at least 4 sequential reports
# then...filter down to first 4 reports in that sequence
all_data_First4 = all_data >> \
                        inner_join(CRN_with_Sufficient_Reports, by='crn') >> \
                            rename(report_date = X.report_date_x) >> \
                                inner_join(UniqueCRN_Date_Count, by=['crn', 'report_date']) >> \
                                    rename(sequence = X.sequence_y) >> \
                                        filter_by(X.sequence <= report_yrs)

# then...arrange the data nicely
# but first identify required columns
fields = list(all_data.columns)
# fields.append('sequence')
fields.append('Report_Ref')

all_data_First4 = all_data_First4 >> \
                        arrange(X.crn, X.sequence) >> \
                            select(fields)

# set index
all_data_First4 = all_data_First4.set_index('Report_Ref')

#%%
# TRAIN, VALIDATE, TEST
# split into test and train

def train_test_split_byCompany(data, propns=[0.8, 0.1, 0.1], seed=20190822):

    # propns = [train, valid, test]
    # so 0.7, 0.1, 0.2 means 70% training, 10% validation, 10% test

    # index to column, so it can persist though forthcoming wrangling
    data = data.reset_index('Report_Ref', drop=False)

    # identify unique list of companies:
    companies = data['crn'].unique()

    # get random sequence as long as data
    np.random.seed(seed)
    rand_seqnc = np.random.rand(len(companies))

    # training
    train_mask = rand_seqnc < propns[0]

    # validation
    valid_mask = [True if value < (propns[0] + propns[1]) and value > propns[0] else False for value in rand_seqnc]

    # testing
    testi_mask = rand_seqnc > (propns[0]+propns[1])

    # get filters
    filter_train = pd.DataFrame(data=companies[train_mask], columns=['crn'])
    filter_valid = pd.DataFrame(data=companies[valid_mask], columns=['crn'])
    filter_testi = pd.DataFrame(data=companies[testi_mask], columns=['crn'])

    # return filtered data
    data_train = data >> inner_join(filter_train, by='crn')
    data_train = data_train.set_index('Report_Ref')

    data_valid = data >> inner_join(filter_valid, by='crn')
    data_valid = data_valid.set_index('Report_Ref')

    data_testi = data >> inner_join(filter_testi, by='crn')
    data_testi = data_testi.set_index('Report_Ref')

    return data_train, data_valid, data_testi

#%%

train_data, valid_data, testi_data = train_test_split_byCompany(data=all_data_First4)

#%%
## Wrangle data to NUMPY 3D for feed to LSTM

def to_3D(data, report_yrs=report_yrs):

    # convert to numpy and reshape to 3D [samples, timesteps, features]
    shape_3D = (len(data['crn'].unique()),   # samples (ie companies)
                report_yrs,                  # timesteps (ie reports per company)
                len(data.columns)-2)         # features per report, less 'crn' and 'report_date' (superfluous)

    # drop superfluous columns
    data = data.drop(columns=['crn', 'report_date'])

    # reshape and set to float32 for Tensorflow (otherwise default is float64)
    data_3D = np.array(data).reshape(shape_3D).astype('float32')

    return data_3D

    ## SHAPE TEST
    ## Lets ensure we built that array right, using some simple example data
    ## 12 reports from 3 companies with 10 features per report. First in simple 2D table:
    # sequence_np = np.array(range(120)).reshape((12,10))
    ## Now reshape into 3D array for 3 companies, 4 reports per company, 10 features per report
    # sequence_np.reshape((3,4,10))

#%%

### Convert to 3D Numpy ###

train_np = to_3D(train_data)
valid_np = to_3D(valid_data)
testi_np = to_3D(testi_data)

#%%

# Quick refresher on LSTM (its been over a year since I used them...)
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# and this example of a LSTM VAE
# https://github.com/twairball/keras_lstm_vae/blob/master/lstm_vae/vae.py

# Note the Dense layer for the latent could be wrapped in TimeDistributed()
# This applies the same weights to each input item, as if each is in sequence

#%%
### THE MODEL ###
# inspired by code at https://machinelearningmastery.com/lstm-autoencoders/

# get data shape
timesteps_in   = train_np.shape[1] 
features       = train_np.shape[2] 
LSTM_lyr_size1 = timesteps_in * features # or 2 x features as rule of thumb
LSTM_lyr_size2 = LSTM_lyr_size1 // 2
LSTM_lyr_size3 = LSTM_lyr_size2 // 2

dense_lyr_size = LSTM_lyr_size3 * timesteps_in // 2
latent_dim     = 6 # 3 timesteps, so 3x previous latent dims. 3x2=6

#%%
### Define encoder ###

input_e = Input(shape=(timesteps_in, features), dtype='float32')
encoded = LSTM(LSTM_lyr_size1, activation='relu', return_sequences=True)(input_e)
encoded = LSTM(LSTM_lyr_size2, activation='relu', return_sequences=True)(encoded)
encoded = LSTM(LSTM_lyr_size3, activation='relu', return_sequences=True)(encoded)

# we intend to encode the whole sequence into one point in the latent space
# so we flatten + dense
# But if wanted to apply the densed layer to each timestep individually, we would have used TimeDistributed(Dense), no flatten
encoded = Flatten()(encoded) # dim = (batch_size, 132)

# two dense layers because output from flateen is so large, 132. 
# Need at least two steps to compress down to latent dims
encoded = Dense(dense_lyr_size, activation='relu')(encoded)

# the second dense layer gives us the latent space
# This is a VAE, so we need two params for each latent dimension
# first a mean, second a variance
# Intentionally NO activation
latent_vae = Dense(2 * latent_dim)(encoded)

# For the VAE we split that dense layer into two tensors
# one represents the mean, the other represents the log of the variance
z_mean, z_log_var = tf.split(latent_vae, num_or_size_splits = 2, axis = 1)

# For the MMD we use a different dense layer (not 2x latent), still no activation
# No split will be necessary
latent_mmd = Dense(latent_dim)(encoded)

# VAE Encoder Model
encoder_vae = Model(input_e, [z_mean, z_log_var])
# MMD Encoder Model
encoder_mmd = Model(input_e, latent_mmd)


#%%
# Lets use Eager Execution to test it...
batch_size_test = 5

# VAE encoder test
en_test_vae = encoder_vae(train_np[0:batch_size_test,:,:])

# MMD encoder test
en_test_mmd = encoder_mmd(train_np[0:batch_size_test,:,:])

print("Expecting a batch tensor shape of : (", batch_size_test,",", latent_dim,")")
print("  VAE latent tensor shape:", en_test_vae[0].shape) # Note, VAE outputs a list, so select first item in list as example
print("  MMD latent tensor shape:", en_test_mmd.shape)

#%%
## Define decoder as separate model
# Remember, we aim to reconstruct the entire sequence for the company

input_d = Input(shape=(None, latent_dim), dtype='float32')

decoded = Dense(dense_lyr_size, activation='relu')(input_d)

decoded = Reshape((timesteps_in, int(dense_lyr_size/timesteps_in)))(decoded) # batch size is implied, no need to include it as None.

decoded = LSTM(LSTM_lyr_size3, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(LSTM_lyr_size2, activation='relu', return_sequences=True)(decoded)
decoded = LSTM(LSTM_lyr_size1, activation='relu', return_sequences=True)(decoded)

# for the decoder we need to rebuild the sequence, so need timedistributed dense layers
# one for the values, which use mse loss, and another for the binaries, which use bxent loss
output_regres = TimeDistributed(Dense(12, activation='linear',  kernel_initializer='glorot_normal'))(decoded)
output_binary = TimeDistributed(Dense(10, activation='sigmoid', kernel_initializer='glorot_normal'))(decoded)

# Decoder Model
decoder = Model(input_d, [output_regres, output_binary])

#%%
# Lets use Eager Execution to test it...

# VAE decoder test
de_test_regres_vae, de_test_binary_vae = decoder(en_test_vae[0].numpy())

# MMD decoder test
de_test_regres_mmd, de_test_binary_mmd = decoder(en_test_mmd.numpy())

print("Expecting batch tensor shapes: Regres=(", batch_size_test,",",timesteps_in,", 12). Binary=(", batch_size_test,",",timesteps_in,", 10)")
print("  VAE output tensor shapes: Regres=", de_test_regres_vae.shape, ". Binary=", de_test_binary_vae.shape)
print("  MMD output tensor shapes: Regres=", de_test_regres_mmd.shape, ". Binary=", de_test_binary_mmd.shape)

#%%
### SAMPLING THE LATENT SPACE FOR VAE

# We can't easily use the MMD because we have 6 latent dimensions, not 2
# so, we'll use the VAE architecture

# VAE needs us to sample the latent space, also known as the 'reparamterization' trick

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
    z_sample = epsilon * tf.math.exp(z_log_var * 0.5) + z_mean

    return z_sample

#%%
### HELPER FOR VAE LOSS FUNCTION

def normal_log_likelihood(z_sample, z_mean, z_log_var, reduce_axis = 1):
    
    loglik = tf.constant(0.5, dtype=tf.float32) * (
                tf.math.log(2 * tf.constant(math.pi, dtype=tf.float32)) +
                z_log_var +
                tf.math.exp(-z_log_var) * tf.square((z_sample - z_mean)) )

    return -tf.reduce_sum(loglik, axis = reduce_axis)

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

## setup data, epochs, batches and optimiser
num_epochs = 200
batch_size = 10 # The number of companies. 100 is preferable
optimizer  = tf.keras.optimizers.Adam(1e-4)
filename   = filename = 'LSTM_VAE'
batches_per_epoch = math.ceil(train_np.shape[0]/batch_size)

# prep pandas file to receive training history
# Not yet implemented:
#   Since we use our own custom training loop we need to implement a validation test each epoch
#   This would be recorded in the history with a column: 'train_or_valid'. =0 for train, =1 for validation
history = pd.DataFrame(columns=['epoch','total_loss','loss_nll_total','loss_mmd_total'], dtype=float)

#%%
### TRAINING LOOP VAE

### SUBSEQUENT CHUNK IS A TRAINING LOOP FOR MMD 

# loop over epochs
for epoch in range(num_epochs):

    # initialise loss params for each epoch
    total_loss    = 0
    logpx_z_total = 0
    logpz_total   = 0
    logqz_x_total = 0

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

        # get batch, all timesteps, all features
        x = train_np[batch_start:batch_end,:,:]

        # The forward pass is recorded by a GradientTape, 
        # and during the backward pass we explicitly calculate gradients 
        # of the loss with respect to the model’s weights. 
        # These weights are then adjusted by the optimizer.
        with tf.GradientTape(persistent = True) as tape:

        #### RECORD FORWARD PASS ####

            # Encode the batch
            z_mean, z_log_var = encoder_vae(x)

            # Sample latent space
            z_sample = reparameterise(z_mean, z_log_var)

            # Decode the batch of samples
            preds = decoder(z_sample) # remember preds is a list of two outputs
            
        #### CALCULATE ELBO LOSS FOR VAE ####

            # Following Google Colab, we calculate the Evidenced Lower Bound (Loss) as:
            # ELBO batch estimate = log p(xbatch|zsampled) + log p(z) − log q(zsampled|xbatch)
            # see https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb

        # 1st Loss Component: +log p(xbatch|zsampled)
            # Which is the loglikelihood of the reconstructed samples given 
            # values sampled from latent space. This is a form of 'reconstruction loss'
            crossentropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                        logits = tf.concat(values=preds, axis=2),
                                        labels = x)

            logpx_z = - tf.reduce_sum(crossentropy_loss)
            
            # FYI, a straight forward reconstruction loss would be as follows
            # recon_loss = - tf.reduce_mean(tf.square(x -  tf.concat(values=preds, axis=2)))        
            # but we need a log likelihood, so use above as an approximation.

         # 2nd Loss Component: +log p(z)
            # Which is the prior loglikelihood of z
            # The prior is assumed to be standard normal, most common for VAE
            logpz = normal_log_likelihood(  z_sample  = z_sample,
                                            z_mean    = tf.constant(0, dtype=tf.float32),
                                            z_log_var = tf.constant(0, dtype=tf.float32))

        # 3rd Loss Component: -log q(zsampled|xbatch)
            # Which is the loglikelihood of the samples z given mean and variance 
            # computed from the observed samples x
            logqz_x = normal_log_likelihood(z_sample  = z_sample, 
                                            z_mean    = z_mean, 
                                            z_log_var = z_log_var)

        # Mean of the loss chunks into the ELBO batch estimate:
            # = MEAN(log p(xbatch|zsampled) + log p(z) − log q(zsampled|xbatch))
            loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
               
        # End of Gradient Tape, Forward prop has ended for the batch

        # Sum losses for batches so far 
        # will divide by batch qty at end of epoch to get mean loss per batch
        total_loss    = total_loss + loss
        logpx_z_total = tf.reduce_mean(logpx_z) + logpx_z_total
        logpz_total   = tf.reduce_mean(logpz)   + logpz_total
        logqz_x_total = tf.reduce_mean(logqz_x) + logqz_x_total
        
        # now calculate gradients for encoder
        encoder_gradients = tape.gradient(loss, encoder_vae.variables)
        grads_and_vars_enc = list(zip(encoder_gradients, encoder_vae.variables))

        # and gradients for decoder
        decoder_gradients = tape.gradient(loss, decoder.variables)
        grads_and_vars_dec = list(zip(decoder_gradients, decoder.variables))

        # apply the gradients to the ENCODER weights+biases using the optimiser's learning rate      
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_enc,
                                  name = 'apply_encoder_grads')

        # apply the gradients to the DECODER weights+biases using the optimiser's learning rate       
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_dec,
                                  name = 'apply_decoder_grads')

    # end of batch loop

    # get mean loss per batch (nice to have a big batch size when calculating loss per batch)
    total_loss_perbatch = round(total_loss.numpy()/batches_per_epoch, 10)
    recon_loss_perbatch = round(logpx_z_total.numpy()/batches_per_epoch, 10)

    # Could execute model on validation data here
    
    # Save epoch's training results to pandas file
    losses_train = [[epoch, total_loss_perbatch, recon_loss_perbatch]]
    history = history.append(pd.DataFrame(data=losses_train, columns=['epoch','total_loss','recon_loss']), ignore_index=True, sort=False)

    # every few epochs
    if epoch % 10 == 0 or epoch == num_epochs: #every tenth epoch

        # save snapshot of latent space using validation data
        temp_name = os.path.join(proj_root, *subfolders, 'latents_by_epoch', filename+'_latents_epoch'+str(epoch)+'.npy')
        latents   = encoder_vae(valid_np)
        np.save(temp_name, latents)

        # print losses to screen
        print( 'Epoch {}:'.format(epoch),
            ' total: ',      total_loss_perbatch,     
           ', recon_loss: ', recon_loss_perbatch)
        
    # end of epoch loop

#%%
## Save Model Function

from contextlib import redirect_stdout

def save_model_custom(model_object, filename, proj_root=proj_root, subfolders=subfolders):
    
    # save keras model (ie serialize and send to file)
    model_object.save(os.path.join(proj_root, *subfolders, filename+'_model.h5'), save_format='tf')

    # save weights
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
# Save models, weights and training history

# Subclassed models cannot be saved in tf.keras because
# such models are defined via the body of a Python method, which isn't safely serializable 
# So we can only save the weights
history.to_csv(os.path.join(proj_root, *subfolders, filename+'_history.csv'), mode='w', header=True, index=False)

# Save encoder
save_model_custom(model_object = encoder_vae,
                  filename     = filename+'_enc')

# save decoder
save_model_custom(model_object = decoder,
                  filename     = filename+'_dec')

#%%
### Prep for MMD

filename = 'LSTM_MMD'
subfolders = ['SaveModels_AutoencLSTM_MMD']

#%%
### TRAINING LOOP MMD

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

        # get batch, all timesteps, all features
        x = train_np[batch_start:batch_end,:,:]

        # The forward pass is recorded by a GradientTape, 
        # and during the backward pass we explicitly calculate gradients 
        # of the loss with respect to the model’s weights. 
        # These weights are then adjusted by the optimizer.
        with tf.GradientTape(persistent = True) as tape:

        #### RECORD FORWARD PASS ###
            mean  = encoder_mmd(x)
            preds = decoder(mean)

        #### CALCULATE MMD LOSS ####

            # generate some random gaussians to compare with real values 
            true_samples = tf.random.normal(
                                shape = [batch_size, latent_dim],
                                mean  = 0.0,
                                stddev= 1.0,
                                dtype = tf.float32)
            
            # compute the MMD distance between the random Gaussians and real samples
            loss_mmd = compute_mmd(true_samples, mean)

            # compute a reconstruction loss
            loss_nll = tf.reduce_mean(tf.square(x - tf.concat(values=preds, axis=2)))

            # batch loss = reconstruction loss + MMD loss
            loss = loss_nll + loss_mmd
               
        # End of Gradient Tape, Forward prop has ended for the batch

        # Sum losses for batches so far 
        # will divide by batch qty at end of epoch to get mean loss per batch
        total_loss     = loss     + total_loss
        loss_mmd_total = loss_mmd + loss_mmd_total
        loss_nll_total = loss_nll + loss_nll_total
        
        # now calculate gradients for encoder
        encoder_gradients  = tape.gradient(loss, encoder_mmd.variables)
        grads_and_vars_enc = list(zip(encoder_gradients, encoder_mmd.variables))

        # and gradients for decoder
        decoder_gradients  = tape.gradient(loss, decoder.variables)
        grads_and_vars_dec = list(zip(decoder_gradients, decoder.variables))
        
        # apply the gradients to the ENCODER weights+biases using the optimiser's learning rate      
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_enc,
                                  name = 'apply_encoder_grads')

        # apply the gradients to the DECODER weights+biases using the optimiser's learning rate       
        optimizer.apply_gradients(grads_and_vars = grads_and_vars_dec,
                                  name = 'apply_decoder_grads')

    # end of batch loop

    # get mean loss per batch (nice to have a big batch size when calculating loss per batch)
    total_loss_perbatch     = round(    total_loss.numpy()/batches_per_epoch, 10)
    loss_nll_total_perbatch = round(loss_nll_total.numpy()/batches_per_epoch, 10)
    loss_mmd_total_perbatch = round(loss_mmd_total.numpy()/batches_per_epoch, 10)

    # Could execute model on validation data here

    # Save epoch's training results to pandas file
    losses_train = [[epoch, total_loss_perbatch, loss_nll_total_perbatch, loss_mmd_total_perbatch]]
    history = history.append(pd.DataFrame(data=losses_train, columns=['epoch','total_loss','loss_nll_total','loss_mmd_total']), ignore_index=True, sort=False)

    # every few epochs
    if epoch % 10 == 0 or epoch == num_epochs: #every tenth epoch

        # save snapshot of latent space using validation data
        temp_name = os.path.join(proj_root, *subfolders, 'latents_by_epoch', filename+'_latents_epoch'+str(epoch)+'.npy')
        latents   = encoder_mmd(np.array(valid_np))
        np.save(temp_name, latents)

        # print losses to screen
        print( 'Epoch {}:'.format(epoch),
            ' total: ',      total_loss_perbatch,
           ', recon_loss: ', recon_loss_perbatch)
        
    # end of epoch loop

#%%
# Save models, weights and training history

# Subclassed models cannot be saved in tf.keras because
# such models are defined via the body of a Python method, which isn't safely serializable 
# So we can only save the weights
history.to_csv(os.path.join(proj_root, *subfolders, filename+'_history.csv'), mode='w', header=True, index=False)

# Save encoder
save_model_custom(model_object = encoder_mmd,
                  filename     = filename+'_enc')

# save decoder
save_model_custom(model_object = decoder,
                  filename     = filename+'_dec')



