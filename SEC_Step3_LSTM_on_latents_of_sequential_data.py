#%%
### AI FOR BUSINESS PLANS: LSTM AUTOENCODERS

# So far we have trained a VAE to give us a latent space of company reports.
# But many companies present us with a time series of company reports, 
# not just one-off reports.

# We have built a VAE with a latent space of company reports, 
# We can witness the trajectory of companies through this latent space as time passes

# Indeed, we can train a time series model on those trajectories in latent space.
# Such a model could then generate forecasts of the next step in the sequence
# and the latent space could be used for generating 'business plans', ie probable trajectories


#%%
#### IMPORTS
import os
import cProfile
import re

import math       as math
import pandas     as pd
import numpy      as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras           import Model
from tensorflow.keras.layers    import Input, LSTM, Dense, Lambda, LeakyReLU, BatchNormalization, Concatenate, Reshape, RepeatVector, TimeDistributed, Flatten
from tensorflow.keras.models    import load_model, save_model
from tensorflow.keras.utils     import plot_model
from tensorflow.keras.callbacks import EarlyStopping

#%% 
#### Confirm environment

print("Tensorflow Version:",               tf.__version__)
print("Is tensorflow executing eagerly? ", tf.executing_eagerly())
print("Is tensorflow using GPU? ",         tf.test.is_gpu_available())
print("Is tensorflow using Cuda? ",        tf.test.is_built_with_cuda())

#%%
### LOAD DATA

proj_root  = 'E:\\Data\\SEC'
subfolders = ['SaveModels_LSTM_on_VAE']

sds_10cols     = pd.read_csv(os.path.join(proj_root, 'sds_10cols.csv'    ), index_col=0)
means_10cols   = pd.read_csv(os.path.join(proj_root, 'means_10cols.csv'  ), index_col=0)

x_train_10cols = pd.read_csv(os.path.join(proj_root, 'x_train_10cols.csv'), index_col='Report_Ref')
x_valid_10cols = pd.read_csv(os.path.join(proj_root, 'x_valid_10cols.csv'), index_col='Report_Ref')
x_testi_10cols = pd.read_csv(os.path.join(proj_root, 'x_testi_10cols.csv'), index_col='Report_Ref')

cols_regres= 11
cols_binary=  9


#%%
### DATA WRANGLING ###

# Our time series model will need to be trained on sequences of reports
# At the very least we need four sequential reports for each company
# three to form a sequence and the fourth to be our predicted report
# SEC data is quarterly, so we'll find most sequences have timesteps of 3months

# Since each sequence must be for a given company...
# ...step 1 is to extract the company registration number from the Report_Ref (crn=cik+coreg)
# of the above data sets (train, test and validation)
def get_unique_cos(df):
    df['crn'] = df.index.str.extract(r'(-[0-9]*_[^_]*_)').iloc[:,0].tolist()
    unique_cos = df['crn'].unique()
    return unique_cos

unique_cos_train = get_unique_cos(x_train_10cols)
unique_cos_valid = get_unique_cos(x_valid_10cols)
unique_cos_testi = get_unique_cos(x_testi_10cols)

#%%
# The training data will eventually be a 3D tensor [samples, timesteps, features]
# Where each series of reports has 4 timesteps, each being a report 3months after the previous

# This will be easiest done by operating on the entire data (trian+valid+test) at once
# then splitting out by train, test, valid . So here we unify all data into one dataframe:

all_data = pd.concat([x_train_10cols, x_testi_10cols, x_valid_10cols])

#%% 
# Any duplicates in the Report_Ref ?
print( len(all_data[all_data.index.value_counts() > 1] ))

#%%
## No duplicates, great, let's continue...

## Reconfigure data into sequences for each company
# the index is company id and date of report, lets get those into columns of their own
# get unique report crn and date combinations
all_data['crn'] = all_data.index.str.extract(r'(-[0-9]*_[^_]*_)').iloc[:,0].tolist()

# extract the report date from the Report_Ref
all_data['report_date'] = all_data.index.str.extract(r'(_[1,2][9,0,1,2][0-9][0-9][0,1][0-9][0-3][0-9]_)').loc[:,0].tolist()
all_data['report_date'] = all_data['report_date'].str[1:9].astype('int')
all_data['report_date'] = pd.to_datetime(all_data['report_date'], format='%Y%m%d', errors='ignore')

UniqueCRN_Date = all_data[['crn', 'report_date']].drop_duplicates(['crn', 'report_date']).sort_values(['crn', 'report_date'])

# Get chronological sequence number for each report for a given company; 1,2,3,4 etc
# Step1. Get sequence using pandas equivalent of ROW_NUMBER() OVER(ORDER BY report_date PARTITION BY crn)
# Also get time between reports, may have skipped a year, or be much less than a year
UniqueCRN_Date = UniqueCRN_Date.assign(
                                    sequence = UniqueCRN_Date.sort_values(['report_date'], ascending=True)
                                                .groupby(['crn'])
                                                .cumcount() 
                                            + 1).sort_values(['crn','report_date','sequence'])

# Calculate interval between dates, some reports issued annually, some quarterly, some erratically
# get previous date in sequence
UniqueCRN_Date['report_date_lag'] = UniqueCRN_Date.groupby(['crn'])['report_date'].shift(1)

# get date difference
UniqueCRN_Date['diff_months'] = UniqueCRN_Date['report_date'] - UniqueCRN_Date['report_date_lag']

# handle errors (NaT) for each new sequence of reports (ie no previous date)
UniqueCRN_Date['diff_months'] = UniqueCRN_Date['diff_months'].fillna(pd.Timedelta(days=0))

# round differences to integers of months
UniqueCRN_Date['diff_months'] = round(UniqueCRN_Date['diff_months'] / np.timedelta64(1,'M'),0).astype(int)

# Flag that time difference is same (1) or different (0)
# first get lag of diff_months
UniqueCRN_Date['diff_months_lead'] = UniqueCRN_Date.groupby(['crn'])['diff_months'].shift(-1)
UniqueCRN_Date['diff_months_lead'] = UniqueCRN_Date['diff_months_lead'].fillna(0).astype(int)

# if diff_months is zero then use next diff_months in sequence
def same_or_diff(row):
    
    if row['diff_months'] == 0:
        returnval = row['diff_months_lead']
    else :
        returnval = row['diff_months']

    return returnval

UniqueCRN_Date['diff_months_calc'] = UniqueCRN_Date.apply(lambda row: same_or_diff(row), axis=1)
UniqueCRN_Date = UniqueCRN_Date.drop(columns=['diff_months','diff_months_lead', 'report_date_lag'])

# Now we can get continuous sequences of reports where diff_months remains same (eg always 3mths apart)
UniqueCRN_Date = UniqueCRN_Date.assign(
                    sequence2 = UniqueCRN_Date.sort_values(['report_date'], ascending=True)
                                .groupby(['crn', 'diff_months_calc'])
                                .cumcount() 
                                + 1).sort_values(['crn','report_date'])

# But what interval between reports should we choose?
# Some companies report annually, others quarterly, others erratically.

#%%

# Let's count the companies who have at least 4 reports
# and group by reporting interval ('diff_months_calc')
seq_length = 4

candidate_co = UniqueCRN_Date[UniqueCRN_Date['sequence2'] == seq_length]
candidate_co = candidate_co.groupby(['diff_months_calc'])['crn'].nunique()

print(candidate_co)

#%%

# So the vast majority of companies are making quarterly reports (every 3 months)
# We need the Report_Ref for all companies which have > 4 sequential reports
# where the reporting_interval is 3mths

# get max sequence over each crn/reporting_interval combination.
UniqueCRN_Seq2Max = UniqueCRN_Date.groupby(['crn','diff_months_calc'])['sequence2'].max().reset_index(drop=False)

# filter to where the max sequence length is >= 4 and the reporting interval is 3mths
UniqueCRN_Seq2Max_4over = UniqueCRN_Seq2Max[(UniqueCRN_Seq2Max['sequence2'] >=4) &
                                            (UniqueCRN_Seq2Max['diff_months_calc'] == 3) ]

# filter the list of reports by the above list of companies and reporting intervals
# using a join (annoyingly named 'merge' in pandas) to achieve this.
candidate_ReportRef = UniqueCRN_Date.reset_index(drop=False).merge(
                            right = UniqueCRN_Seq2Max_4over, 
                            on    = ['crn', 'diff_months_calc'], 
                            how   = 'inner')

# filter the underlying data by the list of selected Report_Ref's
all_data_selected = all_data.merge(
                            right = candidate_ReportRef[['Report_Ref', 'sequence2_x', 'sequence2_y']],
                            on    = ['Report_Ref'],
                            how   = 'inner').sort_values(by=['crn', 'report_date'])

# apply more useful names
all_data_selected = all_data_selected.rename(
                            columns={"sequence2_x": "sequence", 
                                     "sequence2_y": "sequence_max"})

# how many reports do we have for training and testing?
print("Qty of reports for training and testing = ", len(candidate_ReportRef))

#%%

# Now we'll split this seq2seq data into sets for train, test, validate
# we'll use the same companies as were in the original split, which was will be used for training the VAE
# So the training companies for the VAE are the training companies for the sequence model
# This will allow robus end to end training, through both models

x_train_seq2seq = all_data_selected[all_data_selected['crn'].isin(unique_cos_train)]
x_valid_seq2seq = all_data_selected[all_data_selected['crn'].isin(unique_cos_valid)]
x_testi_seq2seq = all_data_selected[all_data_selected['crn'].isin(unique_cos_testi)]

print("Train rows: ", len(x_train_seq2seq))
print("Valid rows: ", len(x_valid_seq2seq))
print("Test rows : ", len(x_testi_seq2seq))

#%%

# The LSTM will be given three company reports and be asked to forecast one
# so we need to convert our data into sequences, presented in a 3D tensor 
# [samples, timesteps, features]

# Now, we have as many sequences than we have reports, which is good news.
# Imagine nine reports in a sequence, A, B, C, D, E, F, G, H, I
# The sequences we can feed to our LSTM are:
# input = A, B, C. Output = D
# input = B, C, D. Output = E
# input = C, D, E. Output = F
# input = D, E, F. Output = G
# input = E, F, G. Output = H
# input = F, G, H. Output = I

# In effect we are sliding a window across the data and repeating the items in the window
# until such time as the window overlaps the data for another company
# The below code is a function to output all the sequences within the data
# a subsequent function will reshape the data into a 3D tensor

from io import StringIO
from csv import writer 

def get_sequences(source_df, filename, seq_length=seq_length):

    source_df['sequence_grp'] = 1
    col_ref_grp = list(source_df.columns).index('sequence_grp')
    output      = StringIO()
    csv_writer  = writer(output)

    # for each row...
    for i in range(0,len(source_df)):

        # give hope to user!
        if i % 2000 == 0:
            print("Percent complete: ", i/len(source_df)*100)

        # get window
        # BEWARE - WE MUST USE .copy(), 
        # else later operations were on all_data_selected, not on window
        window = source_df.iloc[i:i+seq_length].copy()

        # get first and last row in window
        window_opn = window.iloc[0]
        window_cls = window.iloc[-1]

        # if the window's opening and closing crn's are the same, 
        # then we copy the window into our new record set
        if window_opn['crn'] == window_cls['crn']:
                
            # The sequence of the row becomes the group number
            # given to all records in the window which we will create
            window['sequence_grp'] = window_opn['sequence']

            # Thats it!
            # Now we save the window as the created (aka intermediate) sequence
            # Pandas was never designed for adding thousands of rows efficiently
            # So, as per https://stackoverflow.com/questions/41888080/python-efficient-way-to-add-rows-to-dataframe
            # use csv format, but in memory, to append huge numbers of rows efficiently
            for repindex, window_row in window.iterrows():
                csv_writer.writerow(window_row)

    # we need to get back to the start of the StringIO
    output.seek(0)

    # read the data
    source_df_seqs = pd.read_csv(output, header=None)

    # apply columns and sort
    source_df_seqs.columns = source_df.columns
    source_df_seqs = source_df_seqs.sort_values(by=['crn', 'sequence_grp', 'sequence'])

    # There should always be a set number ('seq_length') of reports within 
    # each sequence_grp in each company
    check = source_df_seqs[['Report_Ref','crn','sequence_grp' ]].groupby(['crn','sequence_grp']).count()
    keep_only = check[check['Report_Ref'] == 4]
    source_df_seqs = pd.merge(source_df_seqs, 
                              keep_only, 
                              on=['crn', 'sequence_grp'], 
                              how='inner',
                              suffixes = ('','_y'))

    # save progress
    source_df_seqs.to_csv(os.path.join(proj_root, filename), sep='\t', mode='w', header=True, index=False)

    print("Complete: ", filename)

    return source_df_seqs


#%%

x_train_sequences_allcols = get_sequences(source_df=x_train_seq2seq, filename='x_train_seq2seq.csv')
x_valid_sequences_allcols = get_sequences(source_df=x_valid_seq2seq, filename='x_valid_seq2seq.csv')
x_testi_sequences_allcols = get_sequences(source_df=x_testi_seq2seq, filename='x_testi_seq2seq.csv')

print("Train rows: ", len(x_train_seq2seq))
print("Valid rows: ", len(x_valid_seq2seq))
print("Test rows : ", len(x_testi_seq2seq))

#%%

# select only the required columns
# ie exclude the columns created purely for calculations and wrangling

selected_cols = [
       'Report_Ref', 
       'sic',
       'ddate' ,
       'Assets', 
       'AssetsCurrent',
       'CashAndCashEquivalentsAtCarryingValue', 
       'Liabilities',
       'LiabilitiesCurrent', 
       'PropertyPlantAndEquipmentNet',
       'RetainedEarningsAccumulatedDeficit', 
       'StockholdersEquity',
       'CommonStockValue', 
       'Assets_ispos', 
       'AssetsCurrent_ispos',
       'CashAndCashEquivalentsAtCarryingValue_ispos', 
       'Liabilities_ispos',
       'LiabilitiesCurrent_ispos', 
       'PropertyPlantAndEquipmentNet_ispos',
       'RetainedEarningsAccumulatedDeficit_ispos', 
       'StockholdersEquity_ispos',
       'CommonStockValue_ispos']

x_train_sequences = x_train_sequences_allcols[selected_cols]
x_valid_sequences = x_valid_sequences_allcols[selected_cols]
x_testi_sequences = x_testi_sequences_allcols[selected_cols]

# change Report_Ref from a column to an index
x_train_sequences = x_train_sequences.set_index('Report_Ref')
x_valid_sequences = x_valid_sequences.set_index('Report_Ref')
x_testi_sequences = x_testi_sequences.set_index('Report_Ref')

#%% Extra data for plotting
# Later we'll want to plot successful businesses vs unsuccessful
# it'll be easier to interrogate the original data for that

# We can simply extract the correct data using the Report_ref as join
# between the original data and our training, validation, testing tables.

def reconstitute(sequences):

    # We execute the data wrangling in reverse order. The result is that we must
    # 1. multiply by SD
    sequences_recon = sequences.to_numpy()[:,0:cols_regres] * sds_10cols.to_numpy().T

    # 2. add mean
    sequences_recon = sequences_recon + means_10cols.to_numpy().T

    # 3. exp() (e to the power)
    sequences_recon = np.exp(sequences_recon)

    # 4. apply correct sign (+ve/-ve)
    # a. move from 0 / 1 representation to -1 / +1 representation
    sequences_posneg = np.where(sequences.to_numpy()[:,cols_regres:] == 0,-1,1)
    # b. apply the multiplication, to get correct sign
    sequences_regres = sequences_recon[:, 2:] * sequences_posneg
    # c. unify the columns of data
    sequences_recon = np.concatenate((sequences_recon[:, 0:2],
                                      sequences_regres),
                                      axis = 1)

    # Convert back to pandas
    sequences_recon = pd.DataFrame(data    = sequences_recon,
                                   columns = sequences.columns[0:cols_regres])
    # apply Report_Ref as index
    sequences_recon.index = sequences.index

    return sequences_recon

#%%
# Now apply a trajectory value across the four reports, (end - start) / start
# we'll use this as a colour range, green for growing, red for shrinking

def add_growth_col(data_sequences, data_allcols):

    # use above function to reconstitute values from exponents to real values
    data_recon = reconstitute(data_sequences)

    # add the sequence and sequence_grp columns back into our data (they were removed in above chunk)
    data_recon = pd.concat([data_recon, 
                            pd.DataFrame(data_allcols.set_index('Report_Ref')[['crn', 'sequence_grp', 'sequence']])],
                            ignore_index=False,
                            sort=False,
                            axis=1)

    # get a column with the sequence within each sequence grp, 1,2,3,4 then 1,2,3,4
    data_recon = data_recon.assign(
                                sequence_crn = data_recon.sort_values(['crn', 'sequence_grp', 'sequence'], ascending=True)
                                                .groupby(['crn', 'sequence_grp'])
                                                .cumcount()
                                                + 1).sort_values(['crn','sequence_grp', 'sequence'])

    # filter to sequence_grp in 1,4 (start and end)
    data_recon_st = data_recon[data_recon['sequence_crn'].isin([1,4])]

    # shift the start values to the same row as the finish values
    data_recon_st = pd.concat([data_recon_st,
                                data_recon_st[['CommonStockValue']].shift(periods=1)],
                                ignore_index=False,
                                sort=False,
                                axis=1)
    # rename shifted column
    colnames = data_recon_st.columns.to_list()
    colnames[-1] = 'CommonStockValue_start'
    data_recon_st.columns = colnames

    # then remove the start rows
    data_recon_st = data_recon_st[data_recon_st['sequence_crn'] == 4]

    # calculate (end - start) / start for each grp (from report 1 to report 4)
    def calc_growth(row):
        #handle div by zero
        if row['CommonStockValue_start'] == 0:
            returnval = 0
        else :
            diff = (row['CommonStockValue'] - row['CommonStockValue_start'])
            returnval = diff / row['CommonStockValue_start']

        return returnval

    data_recon_st['CommonStockValue_growth_pc'] = data_recon_st.apply(lambda row: calc_growth(row), axis=1)



    # apply to all members (1,2,3,4)
    data_recon = pd.merge(data_recon.reset_index(), 
                          right= data_recon_st[['crn', 'sequence_grp', 'CommonStockValue_start', 'CommonStockValue_growth_pc']], 
                          on   = ['crn', 'sequence_grp'], 
                          how  = 'left',
                          suffixes = ('','_y')).set_index('Report_Ref')

    # Use growth value to create colour range 0-255

    # cap silly values where start was near 0, so growth appears infinite
    # Most growth is <0.25, so we cap it at 0.5 growth or 0.5 shinkage
    data_recon['CommonStockValue_growth_pc'] = data_recon['CommonStockValue_growth_pc'].apply(lambda x: -0.5 if (x < -0.5) else 0.5 if (x > 0.5) else x)
    # min is now -0.5, must shift to 0
    data_recon['CommonStockValue_growth_255'] = data_recon['CommonStockValue_growth_pc'] + 0.5
    # growth is exponentially distributed, to make colour useful, we'll apply log1p
    data_recon['CommonStockValue_growth_255'] = np.log1p(data_recon['CommonStockValue_growth_255'].to_numpy())
    # scale to 0-255
    data_recon['CommonStockValue_growth_255'] = round(data_recon['CommonStockValue_growth_255'] / max(data_recon['CommonStockValue_growth_255']) * 255, 0)

    return data_recon

#%%

## To recap, when we explore our results we'd like
# to view them in the context of some useful parameter, eg growth
# so we created a growth column to the data
# would have been easier if it had been there in the first place!

# apply 'growth column' to training set
x_train_sequences_recon = add_growth_col(data_sequences = x_train_sequences, 
                                         data_allcols   = x_train_sequences_allcols)

#display histogram of growth
x_train_sequences_recon['CommonStockValue_growth_255'].hist()

#%%

# Let's visualise some successful (profitable) trajectories through latent space
# Vs some unsuccessful trajectories

# We'll need to randomly select sequences of reports (1 sequence = 4 reports)
# so create a table to do that
sequence_ids = x_train_sequences_recon[['crn', 'sequence_grp','CommonStockValue_growth_pc', 'CommonStockValue_growth_255']].drop_duplicates()
n_samples = 10

# Randomly select some trajectories where growth was +10% to +20% over the year
sample_gro_hi = sequence_ids[(sequence_ids['CommonStockValue_growth_pc'] > 0.1)
                              & 
                             (sequence_ids['CommonStockValue_growth_pc'] < 0.2)].sample(n=n_samples)

# Randomly select some trajectories where growth was -10% to -20% over the year
sample_gro_lo = sequence_ids[(sequence_ids['CommonStockValue_growth_pc'] < -0.1)
                              & 
                             (sequence_ids['CommonStockValue_growth_pc'] > -0.2)].sample(n=n_samples)

# Extract the sequences matching those ids for growth +10% to +20% over the year
sample_gro_hi = pd.merge(x_train_sequences_allcols.reset_index(),
                         right= sample_gro_hi, 
                         on   = ['crn', 'sequence_grp'], 
                         how  = 'inner').set_index('Report_Ref').sort_values(['crn', 'sequence_grp', 'ddate'])

# Extract the sequences matching those ids for growth -10% to -20% over the year
sample_gro_lo = pd.merge(x_train_sequences_allcols.reset_index(),
                         right= sample_gro_lo, 
                         on   = ['crn', 'sequence_grp'], 
                         how  = 'inner').set_index('Report_Ref').sort_values(['crn', 'sequence_grp', 'ddate'])

# output columns
cols_gro = x_train_sequences.columns.to_list()+['CommonStockValue_growth_255']
sample_gro_lo = sample_gro_hi[cols_gro]
sample_gro_hi = sample_gro_lo[cols_gro]

#%%
## Get VAE latents for those points

# We could feed the training, validation and test data sets directly into an LSTM
# However, this investigation calls for the ability to be generative, 
# so we need to work with the latent space representations of each record.
# Therefore, we load the VAE encoder..
vae_encoder = load_model(os.path.join(proj_root, 'SaveModels_VAE', 
                        'Lossregression_alpha0.0_latent2_batch100_encoder_model.h5'), 
                        compile=False)

# convert to format required by encoder
sample_gro_lo_np = sample_gro_lo.drop(columns=['CommonStockValue_growth_255']).reset_index(drop=True).to_numpy().astype('float32')
sample_gro_hi_np = sample_gro_hi.drop(columns=['CommonStockValue_growth_255']).reset_index(drop=True).to_numpy().astype('float32')

# and project the records into the latent space
sample_gro_lo_latents = vae_encoder(sample_gro_lo_np)
sample_gro_hi_latents = vae_encoder(sample_gro_hi_np)

# The encoder outputs a list, z_mean and z_logvar. 
# We only want z_mean, so take the 0th element.
# Also, we want the data in numpy format, not a tf.Tensor
sample_gro_lo_latents = sample_gro_lo_latents[0].numpy()
sample_gro_hi_latents = sample_gro_hi_latents[0].numpy()

# add the colour back as the third column
sample_gro_lo_latents = np.concatenate((sample_gro_lo_latents, 
                                        sample_gro_lo['CommonStockValue_growth_255'].to_numpy()[:,np.newaxis]), 
                                        axis=1)

sample_gro_hi_latents = np.concatenate((sample_gro_hi_latents, 
                                        sample_gro_hi['CommonStockValue_growth_255'].to_numpy()[:,np.newaxis]), 
                                        axis=1)

sample_gro_lo_latents = pd.DataFrame(data    = sample_gro_lo_latents, 
                                     columns = ['x', 'y', 'colour'])

sample_gro_hi_latents = pd.DataFrame(data    = sample_gro_hi_latents, 
                                     columns = ['x', 'y', 'colour'])

#%%
# Chart trajectories in latent space
# I hate matplotlib so much, its so clunky compared with ggplot2
# so let's try plotnine, which is akin to ggplot

from plotnine import ggplot, geom_point, geom_line, geom_path, geom_segment, aes, labs, geoms
# could use plydata for dplyr work


p = (ggplot(aes(x='x', y='y', color='colour'),
           pd.concat([sample_gro_lo_latents, sample_gro_hi_latents], axis=0))
    + geom_point()
    + geom_path(aes(group = 'colour'), 
                arrow=geoms.arrow(ends  = "last", 
                                  type  = "closed", 
                                  angle = 20,
                                  length= 0.2))
    + labs(title='Latents of Hi and Lo growth sequences', x='DimX', y='DimY')
)
p

#%%

## Discussion of Latent Space

# The chart shows that the position and trajectory of a sequence in latent space 
# cannot easily be related to the change in the business size over that period

# This is not surprising, the latents encode the static information, not change over time
# We could develop a latent space where each point represents a trajectory,
# but at the loss of viewing each report's location in latent space, each reports individual identify would be lost
# This will be explored, as perhaps it is more accurate, but first we will return to
# building a model which forecasts the fourth report in the sequence, having been given the first 3


#%%
## BACK TO MODELLING

# We could feed the training, validation and test data sets directly into an LSTM
# However, this investigation calls for the ability to be generative, 
# so we need to work with the latent space representations of each record.
# Therefore, we load the VAE encoder..
vae_encoder = load_model(os.path.join(proj_root, 'SaveModels_VAE', 
                        'Lossregression_alpha0.0_latent2_batch100_encoder_model.h5'), 
                        compile=False)

# and project the records into the latent space
x_train_latent = vae_encoder(x_train_sequences.to_numpy().astype('float32'))
x_valid_latent = vae_encoder(x_valid_sequences.to_numpy().astype('float32'))
x_testi_latent = vae_encoder(x_testi_sequences.to_numpy().astype('float32'))

# The encoder outputs a list, z_mean and z_logvar. 
# We only want z_mean, so take the 0th element.
# Also, we want the data in numpy format, not a tf.Tensor
x_train_latent = x_train_latent[0].numpy()
x_valid_latent = x_valid_latent[0].numpy()
x_testi_latent = x_testi_latent[0].numpy()

#%%
##
# The LSTM requires the data in a 3D format, [samples, timesteps, features]
# Now is the time to wrangle the latents into that format
# Timesteps = 4
# Features = latent dimensions (typically 2)
# The function will take either a pandas tables, such as the original data
# or a numpy array, such as the latent data

def to_3D(data, seq_length=seq_length):

    #convert to numpy
    data = np.array(x_train_latent)

    # how many fatures are there?
    features = data.shape[1]

    # how many samples are there?
    samples = data.shape[0] / seq_length
    assert samples % 1 == 0, "error in dimension sizes"
    samples = int(samples)

    # convert to numpy and reshape to 3D [samples, timesteps, features]
    shape_3D = (samples,    # samples (ie companies)
                seq_length, # timesteps (ie reports per company)
                features)   # features per report (not many if in latent space!)

    # reshape and set to float32 for Tensorflow (otherwise default is float64)
    data_3D = data.reshape(shape_3D)

    return data_3D

    ## SHAPE TEST
    ## Lets ensure we built that array right, using some simple example data
    ## 12 reports from 3 companies with 10 features per report. First in simple 2D table:
    # sequence_np = np.array(range(120)).reshape((12,10))
    ## Now reshape into 3D array for 3 companies, 4 reports per company, 10 features per report
    # sequence_np.reshape((3,4,10))

#%%

### Convert Latents to 3D Numpy ###
train_np = to_3D(x_train_latent)
valid_np = to_3D(x_valid_latent)
testi_np = to_3D(x_testi_latent)

#%%

# Quick refresher on LSTM (its been over a year since I used them...)
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
# and this example of a LSTM VAE
# https://github.com/twairball/keras_lstm_vae/blob/master/lstm_vae/vae.py

# Note the Dense layer for the latent could be wrapped in TimeDistributed()
# This applies the same weights to each input item, as if each is in sequence

#%%
### THE MODEL OPTIONS ###

# OPTION 1
# LSTM - Sequence prediction

# Use LSTM on the latents to forecast the next report for the sequence
# Advantages: 
#   Uses the latents, so we can see each report in latent space and manipulate it there
#   Which means all reports in the sequence can be generated reports, ie plans and scenarios 
#   need not be real reports
# Disadvantages:
#   Trained on latent space values, much less data available to the model 
#   than training on actual reports.
#   Output is a foreast location in latent space, not a report
#   Need to use the VAE decoder to expand that location into an actual report

# OPTION 2
# LSTM Autoencoder, possibly with global attention

# Use model on actual report data to forecast the next report for the sequence
# Advantages:
#   Much more data available to the model than available via the latents, forecasts likely to be be better
# Disadvantages:
#   The entire sequences is projected onto a point in LSTM latent space
#   Each individual report loses its identity in latent space
#   So, it is a latent space of trajectories, not of reports, which is less useful for planning
#   Also the model is trained on real reports, so not trained to handle generated reports


# CONCLUSION
# We'll explore Option 1, since the objective is a generative tool for business planning


#%%
## OPTION 1. LSTM - Sequence Prediction
# Using latent speace representation of reports as inputs

def create_LSTM(LSTM_sizes_list, timesteps, features):
  
  # input layer
  # expected input data shape: (timesteps, features). batch_size is implied
  input_sequence = Input(shape=(timesteps, features))
  x = input_sequence
  
  # feature extraction
  # params to train = inputs * outputs + outputs (ie weights + biases)
  # RULE OF THUMB for LSTM size
  # 2/3 * (Nodes_in + Nodes_out)

  for LSTM_size in LSTM_sizes_list:
    # The final LSTM must have return_sequences=False
    # Prior stacked LSTM's must have return_sequences=True
    position = LSTM_sizes_list.index(LSTM_size)
    if position+1 == len(LSTM_sizes_list):
        return_sequences = False
    else:
        return_sequences = True
    print(return_sequences)
    x = LSTM(LSTM_size, return_sequences=return_sequences)(x)

  # finally, two dense layers with relu, one node for each output item
  x = Dense(int(LSTM_size/2), activation='relu')(x)
  output = Dense(features, activation='linear')(x)
  
  # output
  model = Model(inputs  = input_sequence, 
                outputs = output)

  return(model)


#%%
## VAE decoder test
features = train_np.shape[2]

# create an encoder
LSTM_latent_model = create_LSTM(LSTM_sizes_list=[16], timesteps=3, features=features)

# pass a chunk of data, say a batch of 5 samples, through the encoder
# careful to pass only 3 timesteps, not all 4
batch_size_test = 5
test_latent_model = LSTM_latent_model(train_np[0:batch_size_test, 0:3, :])

# decoder test
print("Expecting output with shape: (", batch_size_test,", ", features,")")
print("  Actual shape = ", test_latent_model.shape)

#%%
# Let's see a summary of the proposed model...
LSTM_latent_model.summary()

# Note the output is of shape (Batch, features). This is a 2D shape. 
# During training we'll need np.squeeze to force our targets from 3D to 2D
# eg from (Batch, 3 timesteps, features) to (Batch, features). Not (Batch, 1 timestep, features)

#%%

# Train the LSTM model
LSTM_latent_model.compile(optimizer = 'Adam',                # tf.keras.optimizers.Adam(learning_rate=1e-4)
                          loss      = 'mean_absolute_error') # tf.keras.losses.MeanAbsoluteError) 
epochs      = 100
batch_size  =  32
callback_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

LSTM_latent_history = LSTM_latent_model.fit(x               = train_np[:, 0:3, :], 
                                            y               = np.squeeze(train_np[:, 3:4, :]),
                                                              # squeeze is to give shape 2, not 3
                                            batch_size      = batch_size,
                                            epochs          = epochs,
                                            verbose         = 1,
                                            validation_data = (valid_np[:, 0:3, :], 
                                                               np.squeeze(valid_np[:, 3:4, :])),
                                                               # squeeze is to give shape 2, not 3
                                            shuffle         = True,
                                            callbacks       = [callback_es])
# save model                                                               
LSTM_latent_model.save(os.path.join(proj_root,*subfolders, 'LSTM_latent_model.h5'))

#%% 
# View the training history
plt.plot(LSTM_latent_history.history['loss'])
plt.plot(LSTM_latent_history.history['val_loss'])
plt.title('LSTM on Latents of Company Reports, Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.show()


#%%

# The above training is acceptable for now. Let's proceed without further optimisation
# so we can follow the full process for generating a 'business plan'
# from start to finish

# We start by finding a company report to represent our end point, where we want to be
# Make adaptions to the company report as required to make it fit our expected endpoint
# Using the VAE encoder we then project that report into the latent space

# Similarly for the start point



#%%
## OPTION 2. LSTM Autoencoder

## SKETCH ONLY. NOT A COMPLETED MODEL

# inspired by code at https://machinelearningmastery.com/lstm-autoencoders/
# input will be three timesteps
# the fourth timestep will be the target for training

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
