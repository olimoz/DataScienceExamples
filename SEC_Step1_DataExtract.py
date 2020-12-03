
### SEC DATA ####


# Using the UK's Companies House we have acquired just over 50,000 reports in xbrl format from companies house which are of use to our modelling
# But this isn't very many, as compared with the number published in PDF format
# Furthermore, deep learning models always need more data! Certainly our LSTM models found they only
# had hundreds of sequential reports from which to analyse time series of reports

# So we turn to the Securities & Exchange Commission (SEC) in the USA. The SEC does a lot of data extract
# work for us. They boil down quarterly and annual company reports into tab delimited text files available for download
# from https://www.sec.gov/dera/data/financial-statement-data-sets.html
# field descriptions at: https://www.sec.gov/files/aqfs.pdf

# Note, more detailed reports can also be found at:
# https://www.sec.gov/dera/data/financial-statement-and-notes-data-set.html
# field descriptions at: https://www.sec.gov/files/aqfsn_1.pdf

# Returning to the less detailed, summary, reports...
# Each quarter the SEC releases a zip file of four files:

# num.txt   Values from company reports, for all companies and all common features (eg profit, assets, etc)
#           Selecting the features we want then pivoting to make those features into columns
#           would give us a table equal to the CompaniesHouse data we already have
# pre.txt   The PRE data set contains one row for each line of the financial statements tagged by the filer.  
#           No values, just a lis tof the features given by the report.
#           The source for the data set is the “as filed” XBRL filer submissions. 
# sub.txt   A list of the reporting entities, their names, SIC codes, addresses and reporting periods
# tag.txt   All standard taxonomy tags, not just those appearing in submissions to date

# The most important file is therefore num.txt, let's download...

# This code was brought to you whilst listening to Herbie Hancock : Headhunters (1973, like me)

#%%

### IMPORTS ###

import os     as os
import re     as re
import math   as math
import pandas as pd
import numpy  as np
import zipfile
import requests 
import mca # multiple correspondence analysis, used on factors as opposed to one-hot
from dfply import *


proj_root = 'E:\Data\SEC'

#%%

### Download and unzip the data ###
# from
#   https://www.sec.gov/files/dera/data/financial-statement-data-sets/2019q2.zip
# to
#   https://www.sec.gov/files/dera/data/financial-statement-data-sets/2009q1.zip

# Get list of files to download

df_quarters = pd.DataFrame({
    'key'     : [1 for x in range(4)],
    'Quarter' : ['q1','q2','q3','q4']})

df_years = pd.DataFrame({
    'key'   : [1 for x in range(11)],
    'Year'  : [2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009]})

df_merged = pd.merge(df_quarters,  df_years, on='key')

df_merged = df_merged >> unite('filename', 
                                ['Year', 'Quarter'], 
                                sep       = '', 
                                remove    = False,
                                na_action = 'as_string')

df_merged = df_merged.sort_values(['filename'])                                

filelocn = 'https://www.sec.gov/files/dera/data/financial-statement-data-sets/'

os.chdir(proj_root)

for index, row in df_merged.iterrows():

    filename = row['filename']
    
    url = filelocn + filename + '.zip'
    
    try: # download the data and save to disk
        print('Downloading ' + filename)
        response = requests.get(url, stream=True)
        
        with open(filename+'.zip', 'wb') as handle:
            for block in response.iter_content(1024):
                handle.write(block)
        
        try: # if that worked, unzip the file
            print('Unzipping ' + filename)
            zip_ref = zipfile.ZipFile(filename+'.zip', 'r')
            zip_ref.extractall(filename) # extract to folder with name same as filename
            zip_ref.close()
        except:
            print('file unzip error: ',filename)

    except:
        # Throw an error for bad status codes
        response.raise_for_status()
        print('file download error: ',filename)
        pass

#%%

### Helper functions to extract the downloaded files

## Function to get list of folders in parent folder
def get_list_of_folders(parent_folder_name):
    mypath = os.path.join(parent_folder_name)
    folders_list = [f for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath, f))]
    folders_list_df = pd.DataFrame(folders_list, columns=['foldername'])
    folders_list_df['progress'] = 'Pending'
    del folders_list
    return folders_list_df

#%%

### Get list of all downloaded num.txt files
### and concatenate num.txt files into single file of all num data

num_txt_all = None
num_txt     = None

folders_list = get_list_of_folders(proj_root)
loop_count   = 0

for folder in folders_list['foldername']:

    print("opening ",folder)
    filename = os.path.join(proj_root, folder, 'num.txt')
    try:
        # note the SEC files are supposed in UTF8 format
        # BUt we get numerous unicode errors when loading into pandas
        # These errors relate to the footnotes field, which we're not interested in
        # So, we wimply use the most forgiving encoding to allow us to open the file
        # without errors, which is ISO-8859-1
        num_txt = pd.read_csv(filename, sep='\t', encoding = "ISO-8859-1")
    except Exception as inst:
        print(type(inst))    # the exception instance
        print(inst.args)     # arguments stored in .args
        print(inst)          # __str__ allows args to be printed directly,
        pass

    if loop_count > 0:
        print("concatenating ",folder)   
        num_txt_all = pd.concat([num_txt, num_txt_all], axis=0, sort=False)
    else:
        num_txt_all = num_txt

    loop_count+=1
                           
## Save the file of all num_text data into one big file, 20GB!
num_txt_all.to_csv(os.path.join(proj_root, 'num_txt_all.csv'), sep='\t', mode='w', header=True, index=False)

#%%

# Repeat for the sub.txt files
# these contain the unique reference for the company (as opposed to adsh, which is a ref for the report)
# and the SIC codes the company's primary industry, which could be useful

sub_txt_all = None
sub_txt     = None

folders_list = get_list_of_folders(proj_root)
loop_count   = 0

for folder in folders_list['foldername']:

    print("opening ",folder)
    filename = os.path.join(proj_root, folder, 'sub.txt')
    try:
        sub_txt = pd.read_csv(filename, sep='\t', encoding = "ISO-8859-1")
    except Exception as inst:
        print(type(inst))    # the exception instance
        pass

    if loop_count > 0:
        print("concatenating ",folder)   
        sub_txt_all = pd.concat([sub_txt, sub_txt_all], axis=0, sort=False)
    else:
        sub_txt_all = sub_txt

    loop_count += 1
                           
## Save the file of all num_text data into one big file, 20GB!
sub_txt_all.to_csv(os.path.join(proj_root, 'sub_txt_all.csv'), sep='\t', mode='w', header=True, index=False)

#%%
# Append the company CIK code and SIC code from sub_txt to num_txt

num_txt_all = pd.merge( left  = num_txt_all, 
                        right = sub_txt_all[['adsh','cik','sic','period','fp']], 
                        how   = 'left',
                        on    = ['adsh'])

# The data's concept of unique report ref, adsh, would be better if it included:
# coreg, date of submission, qtrs covered by the report.

num_txt_all['Report_Ref'] = num_txt_all['adsh'].astype(str)  + '-' + \
                            num_txt_all['cik'].astype(str)   + '_' + \
                            num_txt_all['coreg'].astype(str) + '_' + \
                            num_txt_all['ddate'].astype(str) + '_' + \
                            num_txt_all['qtrs'].astype(str)

num_txt_all.to_csv(os.path.join(proj_root, 'num_txt_all.csv'), sep='\t', mode='w', header=True, index=False)
                         
#%%

num_txt_all = pd.read_csv(os.path.join(proj_root, 'num_txt_all.csv'), sep='\t')

num_txt_all.head(20)

### Explore num_txt_all ###

# adsh	    EDGAR Accession Number. The 20-character string formed from the 18-digit number assigned by the SEC to each EDGAR submission.
#               First 10 digits are the company who submitted the file (usually an auditor, nbot the company itself)
#               next 2 are the year, final 8 are the report. ADSH is NOT a unique reference for the company, but for the report.
# coreg	    If specified, indicates a specific co-registrant, the parent company, or other entity (e.g., guarantor).  NULL indicates the consolidated entity.
# tag	    The unique identifier (name) for a tag in a specific taxonomy release.
# version	For a standard tag, an identifier for the taxonomy; otherwise the accession number where the tag was defined.
# ddate	    The end date for the data value, rounded to the nearest month end.
# qtrs	    The count of the number of quarters represented by the data value, rounded to the nearest whole number. “0” indicates it is a point-in-time value.
# uom	    The unit of measure for the value.
# value	    The value. This is not scaled, it is as found in the Interactive Data file, but is limited to four digits to the right of the decimal point.
# footnote	The text of any superscripted footnotes on the value, as shown on the statement page, truncated to 512 characters, or if there is no footnote, then this field will be blank.

# How many companies are reporting?
print("Unique companies in data  = ", len(num_txt_all['adsh'].str[0:10].unique()))

# How many companies are reporting 4qtr (entire year) P&L figures?
# Note, Q0 figures are Balance Sheet. Q1-4 figures are P&L
Q4companies = pd.DataFrame(data=(num_txt_all>>filter_by(X.qtrs==4))['adsh'].unique(), columns=['adsh'])
print("Unique companies reporting four quarters = ", len(Q4companies))

# What are the most common units of measure (uom, mostly currencies) ?
currency_series = num_txt_all['uom'].value_counts()[1:50]
currency_series = currency_series.reset_index(drop=False)

# So if there are more than one standard or currency for a given item in a company report, 
# then we use the most common.
# This is a de-duping process and it must be completed else the pivot fails later

### Currency dedupe (where there are one Report_Ref-tag has multiple values due different currencies)

# There are many uom's which are not currencies, analyse those with three capitals only
currency_series['is_curr'] = currency_series.iloc[:,0].str.match(r'(^[A-Z]{3}$)')

# Wrangle to get a dict of currency and it's priority (highest count= highest priority)
currency_series = currency_series[currency_series['is_curr'] & (currency_series['uom'] > 1)]
currency_series = currency_series.reset_index(drop=True)
currency_series['sequence'] = currency_series.index
currency_series = currency_series.set_index('index', drop=False)
currency_series = currency_series['sequence'].to_dict()

# create switch (aka 'case') function using the above dict
def switch_uom(argument):
    switcher = currency_series
    return switcher.get(argument, len(currency_series)) # defaults to max+1

# apply the dict, so where we had currency name, now we have currency priority
num_txt_all['uom_seq'] = num_txt_all['uom'].apply(switch_uom)

# Next create a count for company report tags which have different standards
num_txt_all = num_txt_all.assign(
                    uom_ct = num_txt_all.sort_values(['uom_seq'], ascending=True)
                        .groupby(['Report_Ref', 'tag'])
                        .cumcount() + 1
                    )
                    # below line is nice to have, but slow to calculate
                    #.sort_values(['Report_Ref','tag','uom','version'])

# Cut out the dupes by 'accounting standard', aka 'version'
num_txt_all = num_txt_all[num_txt_all['uom_ct']==1]

#%%

# Let's de-dupe standards next

# first we note that many 'version' names are simply 'us-gaap' with a year appended, 
# eg us-gaap/2018
# We don't care about year in this context, so let's remove all those '/years'
num_txt_all['version'] = num_txt_all['version'].str.split('/', expand=True).iloc[:,0]
standards_series = num_txt_all['version'].value_counts()

# Append the priority of standards to the data. We can hard code this one...
# create switch (aka 'case') function to crreate a priority sequence for 'version' 

def switch_std(argument):
    switcher = {
        "us-gaap": 1,
        "ifrs"   : 2,
        "dei"    : 3
    }
    return switcher.get(argument, 4) # defaults to 4

# apply the dict, so where we had acctg std name, now we have the standard's priority
num_txt_all['version_seq'] = num_txt_all['version'].apply(switch_std)

# create a count for company report tags which have different standards
num_txt_all = num_txt_all.assign(
                    version_ct = num_txt_all.sort_values(['version_seq'], ascending=True)
                        .groupby(['Report_Ref', 'tag'])
                        .cumcount() + 1
                    )
                    # below line is nice to have, but slow to calculate
                    #.sort_values(['Report_Ref','tag','uom','version'])

# cut out the dupes by 'accounting standard', aka 'version'
num_txt_all = num_txt_all[num_txt_all['version_ct']==1]

# tidy up workings
num_txt_all = num_txt_all.drop(columns=['version_seq', 'version_ct', 'uom_seq', 'uom_ct'])

#%%

# confirm whether any duplicates remain for Report_Ref - tag combinations
num_txt_all = num_txt_all.assign(
                    dupe_ct = num_txt_all.sort_values(['Report_Ref', 'tag'], ascending=True)
                                .groupby(['Report_Ref', 'tag'])
                                .cumcount() + 1)

# print the dupes to screen, if any
print(num_txt_all[num_txt_all['dupe_ct']>1])

#%%

# assuming there are no dupes, let's tidy up again:
del num_txt_all['dupe_ct']

# What is the most common reporting tag (aka item)?
report_tags_count = num_txt_all[['Report_Ref', 'tag', 'uom']]

# remove tags measured in 'stock', stock quantities are somewhat arbitrary
report_tags_count = report_tags_count[~report_tags_count['uom'].str.contains('stock', case=False)]

# no longer need uom in this table of tags...
del report_tags_count['uom']

# remove tags which are about shares, stock quantities are somewhat arbitrary
# note we don't take this approach to the word 'stock' 
# because it could mean 'inventory' or 'equity', both of which we wish to include.
report_tags_count = report_tags_count[~report_tags_count['tag'].str.contains('share', case=False)]

report_tags_count = report_tags_count.groupby('tag').count()
report_tags_count = report_tags_count.sort_values(['Report_Ref'], ascending=False)

# let's see the top 30
print(report_tags_count[0:30])


#%%

# A quick inspection reveals that some tags use a currency as uom, 
# whereas other tags in the same report may use 'pure' as their uom
# We need to be sure that each tag uses only one uom method because
# ultimately we will remove uom, each tag will have just one uom, 
# and then all currencies will be converted to USD

# For now, that one uom method could be either currency, 'pure', or other. 
# Note, if currency, then it could be multiple currencies, thats ok.
# So let's get the number of uom's per tag

def switch_uom(argument):
    uom = argument
    if len(re.findall(r'(^[A-Z]{3}$)', argument)) == 1:
        uom = 'Currency'
    return uom

num_txt_all['uom_method'] = num_txt_all['uom'].apply(switch_uom)
tags_uom_ct = num_txt_all[['uom_method', 'tag']].groupby('tag')['uom_method'].nunique()

# identify tags with multiple uom's
tags_uom_remove = list(tags_uom_ct[tags_uom_ct > 1].index)

# how many will be removed?
print("This many tags will be removed: ", len(tags_uom_remove), " from a total of ", len(tags_uom_ct))

#%%
# index is currenty 'tag' name, more convenient of that is a column
report_tags_count = report_tags_count.reset_index(drop=False)

# Assuming the quantity to be removed is a small sacrifice for simplicity
# then let's go ahead and remove those tags from our list
report_tags_count = report_tags_count[~report_tags_count['tag'].isin(tags_uom_remove)]

# save to file for analysis in spreadsheet
# we need to manually compare US and UK tag names, similar but not precisely the same
report_tags_count.to_csv(os.path.join(proj_root, 'report_tags_count.csv'), mode='w', header=True, index=False)

# There are far too many tags, so we'll take just the top 200.
# then filter our data to just these selected tags
SEC_top_features = report_tags_count[0:200]

#%%
#### Currency conversion ####

# The data is presented by SEC in a number of different currencies, as reported by each company
# We need to convert back to USD
# For now, we will ignore inflation in USD , but there is anargument to bringing all values back
# to USD on a given date, a baseline, say 31-12-2010

# Historical currency conversion tables can be found at the Bank of England
#   http://www.bankofengland.co.uk/boeapps/iadb/index.asp?first=yes&SectionRequired=I&HideNums=-1&ExtraInfo=true&Travel=NIxHPx
# Othere sources are available: 
#   https://www.icaew.com/library/subject-gateways/financial-markets/knowledge-guide-to-exchange-rates/historical-exchange-rates
# but BoE offers free tables by month end, exactly what we need

exchange_boe = pd.read_csv(os.path.join(proj_root, 'exchange_boe.csv'), sep=',')
exchange_boe = exchange_boe.set_index(['YearMonth'])

# we don't have all the currencies, only a selection, we';; need to filter out
# records where we don't have currency conversion to USD in the next step
currencies_available = list(exchange_boe.columns[1:])

# add USD to that list, don't need conversion!
currencies_available.insert(0, 'USD')

print("Currency conversion is available for ...", currencies_available)

#%%

# let's focus on the tags we can use...
num_txt_pv_prep = num_txt_all[num_txt_all['tag'].isin(SEC_top_features['tag'])]

# and the currencies or non-currency uom's we can use...
num_txt_pv_prep = num_txt_pv_prep[(num_txt_pv_prep['uom'].isin(currencies_available)) | (num_txt_all['uom_method']!='Currency')]

# we must also remove any future report dates. 
# It has been noted that some tags have a future date, eg 2023 (its now 2019)
# We can't currency convert for those dates and they are estimates anyway...
# We'll also remove anything prior to 2000, as we struggle for currency conversion before then too
num_txt_pv_prep = num_txt_pv_prep[(num_txt_pv_prep['ddate'] < 20190801) & (num_txt_pv_prep['ddate'] > 19991231)]

# Now we have a much smaller data set let's apply that currency conversion
def convert_to_USD(row):

    source_value = row['value']
    currency     = row['uom']
    uom_method   = row['uom_method']

    if uom_method != 'Currency' or currency == 'USD':
        # no currency conversion required
        return_value = source_value

    else:
        # let's convert...
        yearmonth = int(str(row['ddate'])[0:6])
        rate = exchange_boe.loc[yearmonth, currency]
        # we won't stop div 0 error, because if it happens we messed up!
        return_value = source_value / rate
    
    return return_value

num_txt_pv_prep['value_USD'] = num_txt_pv_prep.apply(convert_to_USD, axis=1)

# We'll do a simple pivot to prove the data is clean and there are no dupes

# Paranoid about dupes, so one last check:
num_txt_pv_prep_ct = num_txt_pv_prep[['Report_Ref', 'tag', 'version', 'uom']].groupby(['Report_Ref', 'tag']).count()
qty_of_dupes = len(num_txt_pv_prep_ct[(num_txt_pv_prep_ct['version'] > 1) & (num_txt_pv_prep_ct['uom'] > 1)])
print("There are ", qty_of_dupes, " dupes in the data")

# another method to find dupes in the format we are aiming for...
try:
    pivot_test = num_txt_pv_prep.pivot( index   = 'Report_Ref',
                                        columns = 'tag', 
                                        values  = 'value_USD')
    del pivot_test
except Exception: #catch code errors, not OS errors
    print("There must have been dupes!")

# let's save work to date
num_txt_pv_prep.to_csv(os.path.join(proj_root, 'num_txt_pv_prep.csv'), mode='w', header=True, index=True)

#%%

# load saved data
# num_txt_pv_prep = pd.read_csv(os.path.join(proj_root, 'num_txt_pv_prep.csv'))

# We now pivot the data such that each row is a company report and each column is a key item in that report (Sales, Expenses, Profit, etc)
num_txt_pv = pd.pivot_table(data    = num_txt_pv_prep, 
                            index   = ['Report_Ref', 'sic' ,'ddate', 'qtrs'], # sic, ddate and qtrs will be mad einto columns later. Only Report_Ref is the index
                            columns = ['tag'],
                            values  = 'value_USD',
                            aggfunc = np.mean) # there won't be any aggregation, because proven no dupes

# We'll make year and qtrs into columns, to be included in the model
num_txt_pv['Report_Ref']= num_txt_pv.index.get_level_values(0)
num_txt_pv['sic']       = num_txt_pv.index.get_level_values(1)
num_txt_pv['ddate']     = num_txt_pv.index.get_level_values(2)
num_txt_pv['qtrs']      = num_txt_pv.index.get_level_values(3)

num_txt_pv = num_txt_pv.reset_index(drop=True)
num_txt_pv = num_txt_pv.set_index('Report_Ref')

# bring these three new columns to far left of table, default is to right end
cols = list(num_txt_pv.columns)
cols.insert(0, cols[-3])
cols.insert(1, cols[-2])
cols.insert(2, cols[-1])
del cols[-3:]
num_txt_pv = num_txt_pv[cols]

# save work
num_txt_pv.to_csv(os.path.join(proj_root, 'num_txt_pv.csv'), mode='w', header=True, index=True)

print("The data for the model has shape: ", num_txt_pv.shape)


#%%
################## INSPECT THE DATA ###########################
# how sparse is our matrix? 
# How many rows (ie company reports) have <X% of the data completed
# in other words, let's analyse the quantiles of the matrix

def get_completeness(matrix):
    # first discover which cells in the matrix have data
    Completeness = matrix[matrix.columns[3:]].notna()

    # sum those cells by row (ie by company report)
    Completeness_row = Completeness.sum(axis=1)
    Completeness_row = Completeness_row.reset_index()
    Completeness_row.columns = ['rownumber','cells_completed']

    Completeness_col = Completeness.sum(axis=0)
    Completeness_col = Completeness_col.reset_index()
    Completeness_col.columns = ['colname','cells_completed']

    # how many columns of data are there
    print("\n","Number of columns = ", len(matrix.columns)-3)

    # how many rows of data are there?
    print("\n","Number of rows = ", len(matrix))

    # analyse row completeness by quantiles, divide by column count gives %
    print("\n", "Completeness of rows, as proportion of all columns available: Quantiles")
    print(Completeness_row['cells_completed'].quantile([0.25,0.5,0.75,1])/(len(matrix.columns)-3))

    # analyse column completeness by quantiles, divide by row count gives %
    print("\n", "Completeness of cols, as proportion of all rows available: Quantiles")
    print(Completeness_col['cells_completed'].quantile([0.25,0.5,0.75,1])/len(matrix))

    return Completeness_row, Completeness_col


#%%
################## REDUCE SPARSITY ###########################

# So, the data is fairly sparse, the most complete are only 42% filled.
# This is a very sparse matrix
# The first step is to remove rows and columns the vast majority of which are blank
# Any deep learning system will likely guess '0' for such columns all the time, and be untrainable
num_txt_pv = pd.read_csv(os.path.join(proj_root, 'num_txt_pv.csv'))

#remove qtr 0 as these are reported with very different features than qtrs 1-4
num_txt_pv_not0 = num_txt_pv[num_txt_pv['qtrs']!=0]

Completeness_row, Completeness_col = get_completeness(num_txt_pv_not0)

# First let's remove COLUMNS of data which are hardly ever completed, we want > 10,000 entries, so better than UK companies House
Selected_cols = Completeness_col >> filter_by(X.cells_completed > 10000) >> arrange(X.cells_completed, ascending=False)
num_txt_pv_notsparse = num_txt_pv_not0[['year','qtrs'] + Selected_cols['colname'].tolist()]
Completeness_row, Completeness_col = get_completeness(num_txt_pv_notsparse)

# Now let's remove ROWS of data which are mostly blank entries, we want > 30 features
Selected_rows = Completeness_row >> filter_by(X.cells_completed > 30) >> arrange(X.cells_completed, ascending=False)
num_txt_pv_notsparse = num_txt_pv_notsparse.loc[list(Selected_rows['rownumber']),:]
Completeness_row, Completeness_col = get_completeness(num_txt_pv_notsparse)

#summarise the situation
print("\n","Data dimensions are now: ", num_txt_pv_notsparse.shape)

# save results
# num_txt_pv_notsparse.to_csv(os.path.join(proj_root, 'num_txt_pv_notsparse.csv'), mode='w', header=True, index=False)

#%%
# We can do a bit better than this, we can fully explore the combinaitons of columns
# and the resulting numbe rof complete rows we get.

# Let's say we want X columns of data, whats the most rows of complete data we can get?

from scipy.special import comb
from itertools import combinations

# noting that the first three columns (0,1 and 2) are Report_Ref, years, tag, 
# so not relevant to this selection (they are always complete)

Completeness = num_txt_pv_not0[num_txt_pv_not0.columns[4:]].notna()
Completeness_col = Completeness.sum(axis=0)
Completeness_col = Completeness_col.reset_index(drop=False)
Completeness_col.columns = ['tag','weight']
Completeness_col = Completeness_col.sort_values('weight', ascending=False)

num_txt_pv_not0_binary = num_txt_pv_not0.notna()


#%%

def get_features_vs_rows(completeness_col, data_binary, attempts=10000):

    # initialisation
    results = pd.DataFrame(columns=['Column_Qty', 'Complete_Rows', 'Column_List'])
    results['Column_List'] = results['Column_List'].astype(object)
    result_count = 0

    # loop around the column quantities we wish to explore
    for column_qty in list(range(8,11)):

        # calculate number of combinations possible
        # NB combinations, not permutations. Combinations don't care about column order, neither do we
        
        combination_qty = comb(N = len(completeness_col), # number of items in the bag
                               k = column_qty, # number of items to be taken from the bag
                               exact     = True, 
                               repetition= False)
  
        # create empty list to hold all column combinations
        column_combinations = []

        # seed with our best guess. ie columns listed in order of completeness
        first_guess = sorted(list(completeness_col.sort_values('weight', ascending=False)[0:column_qty]['tag']))
        column_combinations.append(first_guess)
        
        # If there are less than 10,000 combinations then we will calculate all of them
        if combination_qty < attempts:
            for L in range(0, len(completeness_col) + 1):
                for column_list in combinations(completeness_col['tag'], L):
                    # save combination to list
                    column_combinations.append(column_list)
        else:
        # Else we'll randomly sample 10,000 column combinations
        # The sample function will weight columns by the number of rows completed
        # Note, we only sample from the top 25 columns, else possible combinations becomes impossibly large
            
            # get sample. if sample is already in the list, then we'll need to try again!
            while len(column_combinations) < attempts :

                # get sample, weighted by number of rows completed in the column
                column_list = list(completeness_col.sample(n=column_qty, replace=False, weights='weight')['tag'])
                
                # sort column_list alphabetically
                column_list = sorted(column_list)

                # if the sample is not already in the list then we add it
                if not(column_list in column_combinations):
                    column_combinations.append(column_list)

        # report progress
        print("column_qty=", column_qty,". Combinations to explore = ", len(column_combinations) )

        # iterate thru column_lists, getting total number of completed rows   
        column_list_count = 0

        for column_list in column_combinations:

            # calculate number of complete columns
            complete_rows = sum(data_binary[column_list].sum(axis=1) >= column_qty)

            # save results
            results.loc[result_count,'Column_Qty']    = column_qty
            results.loc[result_count,'Complete_Rows'] = complete_rows
            results.loc[result_count,'Column_List']   = column_list

            # report progress
            if column_list_count % 1000 == 0 :
                highest_rows = results[results['Column_Qty'] == column_qty]['Complete_Rows'].max()
                print("Combinations attempted = ", column_list_count,". Highest complete rows = ",highest_rows)
            
            column_list_count += 1
            result_count += 1

    return results

#%%
results = get_features_vs_rows(completeness_col = Completeness_col[0:25], # NB we use only the first 25 columns (ordered by completeness), else it get impossible!
                               data_binary      = num_txt_pv_not0_binary,
                               attempts         = 10000)

# sort results
results = results.sort_values(by=['Column_Qty', 'Complete_Rows'], ascending=[1,0])

# save results to file, they took hours to generate!
results.to_csv(os.path.join(proj_root, 'Complete_Rows_q1to4.csv'), mode='w', header=True, index=False)

# summarise
summary_q1to4 = results[['Column_Qty', 'Complete_Rows']].groupby(['Column_Qty']).max()
columns_list_best_q1to4 = results[results['Complete_Rows'].isin(summary_q1to4['Complete_Rows'])]

# tell us the answer!
print(columns_list_best_q1to4)

#%%

# here's the summary showing the columns chosen, 
# It shows the same basic columns being exgtended as we add features
# which is re-assuring that we found the best combination.

# 8 features	                                # 9 features	                            # 10 features
# 143,318 reports                               # 77,216 reports                            # 53,160 reports
########################################################################################################################################
# CashAndCashEquivalentsPeriodIncreaseDecrease	CashAndCashEquivalentsPeriodIncreaseDecreaseCashAndCashEquivalentsPeriodIncreaseDecrease
# IncomeTaxExpenseBenefit	                    IncomeTaxExpenseBenefit	                    IncomeTaxExpenseBenefit
# NetCashProvidedByUsedInFinancingActivities	NetCashProvidedByUsedInFinancingActivities	NetCashProvidedByUsedInFinancingActivities
# NetCashProvidedByUsedInInvestingActivities	NetCashProvidedByUsedInInvestingActivities	NetCashProvidedByUsedInInvestingActivities
# NetCashProvidedByUsedInOperatingActivities	NetCashProvidedByUsedInOperatingActivities	NetCashProvidedByUsedInOperatingActivities
# NetIncomeLoss	                                NetIncomeLoss	                            NetIncomeLoss
# OperatingIncomeLoss	                        OperatingIncomeLoss	                        OperatingIncomeLoss
# PaymentsToAcquirePropertyPlantAndEquipment	PaymentsToAcquirePropertyPlantAndEquipment	PaymentsToAcquirePropertyPlantAndEquipment
#	                                            GrossProfit	                                GrossProfit
#		                                                                                    ComprehensiveIncomeNetOfTax

# It's disappointing that many SEC reports exclude values as fundamental as Gross Profit
# We had 50,000 records with UK records.
# We would like > 100,000 records to see how the models perform with more data
# so if using q1-4 data we'd have to settle for just 8 features

# One last option to explore, the Q0 reports
# How many complete features and rows can we achieve with those?

#%%
# Remove qtr 0 as these are reported with very different features than qtrs 1-4
num_txt_pv_isq0 = num_txt_pv[ num_txt_pv['qtrs'] == 0 ]

# get those completness stats again...
Completeness = num_txt_pv_isq0[num_txt_pv_isq0.columns[4:]].notna()
Completeness_col = Completeness.sum(axis=0)
Completeness_col = Completeness_col.reset_index(drop=False)
Completeness_col.columns = ['tag','weight']
Completeness_col = Completeness_col.sort_values('weight', ascending=False)

num_txt_pv_isq0_binary = num_txt_pv_isq0.notna()

# analyse those features, how many rows do we get for 8,9,10 features?
results = get_features_vs_rows(completeness_col = Completeness_col[0:30], 
                               data_binary      = num_txt_pv_isq0_binary,
                               attempts         = 3000)
# sort results
results = results.sort_values(by=['Column_Qty', 'Complete_Rows'], ascending=[1,0])

# save results to file, they took hours to generate!
results.to_csv(os.path.join(proj_root, 'Complete_Rows_q0.csv'), mode='w', header=True, index=False)

# summarise
summary_q0 = results[['Column_Qty', 'Complete_Rows']].groupby(['Column_Qty']).max()
columns_list_best_q0 = results[results['Complete_Rows'].isin(summary_q0['Complete_Rows'])]

# tell us the answer!
print(columns_list_best_q0)

columns_list_best_q0.to_csv(os.path.join(proj_root, 'columns_list_best_q0.csv'), mode='w', header=True, index=False)

#%%

# Here's the summary showing the columns chosen, 
# It shows the same basic columns being exgtended as we add features
# which is re-assuring that we found the best combination.

# 8 features	                        # 9 features	                        # 10 features
# 264,619 reports                       # 188,457 reports                       # 160,943 reports
########################################################################################################################################
# Assets	                            Assets	                                Assets
# AssetsCurrent	                        AssetsCurrent	                        AssetsCurrent
# CashAndCashEquivalentsAtCarryingValue	CashAndCashEquivalentsAtCarryingValue	CashAndCashEquivalentsAtCarryingValue
# LiabilitiesAndStockholdersEquity	    LiabilitiesAndStockholdersEquity	    LiabilitiesAndStockholdersEquity
# LiabilitiesCurrent	                LiabilitiesCurrent	                    LiabilitiesCurrent
# PropertyPlantAndEquipmentNet	        PropertyPlantAndEquipmentNet	        PropertyPlantAndEquipmentNet
# RetainedEarningsAccumulatedDeficit	RetainedEarningsAccumulatedDeficit	    RetainedEarningsAccumulatedDeficit
# StockholdersEquity	                StockholdersEquity	                    StockholdersEquity
#           	                        Liabilities                             Liabilities
#                                                                               CommonStockValue


# Q0 reports are much more detailed, 
# We would like > 100,000 records to see how the models perform with more data
# so we can easily do this with 10 features from Q0 reports.

# BUT
# Q0 reports are clearly balance sheet features, not P&L
# Whereas Q1-4 tend to be P&L

#%%
# read data from file
# columns_list_best_q0 = pd.read_csv(os.path.join(proj_root, 'columns_list_best_q0.csv'))

# The result of the above loop is that we can have 
# a) 160,000 company reports if we settle for 10 columns

# NOTE
#'LiabilitiesAndStockholdersEquity' is simply a sum of Liabilities + StockholdersEquity
# so is superfluous, must be removed.

cols = ['Assets', 
        'AssetsCurrent',
        'CashAndCashEquivalentsAtCarryingValue',
        'Liabilities', 
        'LiabilitiesCurrent', 
        'PropertyPlantAndEquipmentNet', 
        'RetainedEarningsAccumulatedDeficit', 
        'StockholdersEquity',
        'CommonStockValue']

num_txt_pv_isq0_col = num_txt_pv_isq0[['sic', 'ddate', 'qtrs'] + cols]
completes = num_txt_pv_isq0_col.notna().sum(axis=1) == (3+len(cols)) # add the three ref columns
num_txt_pv_isq0_col = num_txt_pv_isq0_col.iloc[list(completes)]

print("Length of data = ", len(num_txt_pv_isq0_col))
print("Features = ",['sic', 'ddate', 'qtrs']+list(cols))

# save to file
num_txt_pv_isq0_col.to_csv(os.path.join(proj_root, 'num_txt_pv_isq0_col.csv'), mode='w', header=True, index=True)


#%%
# Now we have our candidate pandas tables let's engage in a little more data exploration
# This uses the fantastic pandas_profiling package, whose use I'd like to document here...

import pandas_profiling

pandas_profiling.ProfileReport(num_txt_pv_isq0_col)

# Result is that lots of the features are skewed
# We'll need to log the data

#%%

# SIC CODES

# SIC Codes for the SEC can be explored at https://www.sec.gov/info/edgar/siccodes.htm

# There are approx 400 SIC code sin the data, with that many factors one hot will be messy. 
# We need a different encoding approach. For categorical encoding options see:
# http://www.willmcginnis.com/2015/11/29/beyond-one-hot-an-exploration-of-categorical-variables/
# The above resource is written by the author of the category-encoders package for python
# https://pypi.org/project/category_encoders/

# Ordinal vs Categorical
# Truly categorical features would be something like town names
# When plotted against each other they would be equidistant
# SIC codes are somewhat ordinal, ie the 'distance' between codes is meaningful. For example...
# 2600 Paper products and 2611 Pulp mills is small, and with good reason
# But SICs are only somewhat ordinal because there are sudden leaps where the first digit changes ...
# 5990 is Retail Stores whereas 6021 is National Commercial Banks

# Retaining Relevant Information
# Whatever scheme is used should aim to retain the information in the SIC code
# We could use PCA or t-SNE could be used to project all the various features down to 2D
# Then use a the mean location of points belonging to each SIC to represent that SIC code
# But this is convoluted and some informaiton will be lost.

# Alternatively, SIC codes could easily be binary encoded. 
# For 400 records we'd need 2 to the power 9 (512), ie 9 features, whereas we only have 1 now.

# CONCLUSION
# Due to their natural ordinality and messy alternatives, we'll leave SIC codes 'as is' for now!


#%%

# Final steps before creating train/valid/test sets...

# Separate out the reference cols (year and report_ref)
# Add columns to indicate whether a value is positive vs negative
# Log+1 the data

def get_value_cols_only(df):
    '''
        Removes non values columns, ie categoricals and strings, from dataframe
        Uses col names to achieve this, must be disciplined about column names!
        Cannot use dtypes because categoricals are 1 or 0, ie dtype=int64, 
        TAKES: dataframe
        RETURNS: dataframe
    '''
    # we can only transform columns which are values, but not string and not categorical
    # remove the report_ref column (string), if relevant, ie code will not error if Report_Ref is in index
    columns = [colname for colname in df.columns if not colname == 'Report_Ref']

    # if there are _ispos colums then remove those...
    columns = [colname for colname in columns if not colname.endswith('_ispos')]

    # if there are _notna columns then remove those...
    columns = [colname for colname in columns if not colname.endswith('_notna')]

    # apply remaining list of columns
    df = df[columns]
    
    return df

def posneg(value):

    if pd.notnull(value) :
        if value >= 0 :
            return 1
        else :
            return 0
    else:
        # we have a different set of categories for nan, here a 0 will suffice
        return 0

def notna(value):

    if pd.notnull(value) :
        return 1
    else :
        return 0

def apply_cat(df, function, suffix):

    # ignore reference columns
    ref_cols  = [colname for colname in df.columns if colname in ['Report_Ref','year','sic', 'ddate', 'qtrs','period']]
    df_to_map = df.drop(columns=ref_cols) 

    # ignore any categoricals already existing
    df_to_map = get_value_cols_only(df_to_map)

    # save those categoricals for later
    df_categs = df.drop(columns=[*df_to_map.columns.to_list(), *ref_cols])

    # apply function to discover category of data
    new_cols  = df_to_map.applymap(function)

    # apply names to new columns
    new_cols.columns = df_to_map.columns + suffix

    # If applying posneg function then replace -ves with +ve value
    if suffix == '_ispos':
        df_to_map = df_to_map.applymap(abs)

    # If applying notna function then replace Nan with 0
    if suffix == '_notna':
        df_to_map = df_to_map.replace(to_replace=np.nan, value=0)

    # append data as new columns (i.e. to right, axis=1) of existing data
    return pd.concat([df[ref_cols], df_to_map, df_categs, new_cols], axis=1)

#%%

# qtrs is always zero, so we will ignore that column
cols_to_keep = [colname for colname in num_txt_pv_isq0_col.columns if colname not in ['qtrs']]
num_txt_pv_isq0_col = num_txt_pv_isq0_col[cols_to_keep]

# apply pos/neg function to the two simple options of data representation which EXCLUDE NA's
sec_cols = apply_cat(num_txt_pv_isq0_col, posneg, '_ispos')
sec_cols.to_csv(os.path.join(proj_root, 'sec_cols.csv'), mode='w', header=True, index=True)

#%%
################## LOG AND SCALE ###########################

# Financial values are often approximately exponentially distributed
# this is because money is poisson distributed, it is a 'count',
# where we have to pass through low numbers to get to high numbers, such is the path to creating wealth!
# So, we will log and then standardise ALL features

from sklearn.preprocessing import FunctionTransformer, StandardScaler

def log_standardise(df):
    '''
        Logs and Standardises (subtract by mean, div by sd) value data (not categoricals)
        Intended for feeding into neural net
        TAKES: dataframe
        RETURNS: dataframe
    '''
    df = sec_cols

    # ensure we log and scale only value fields, not categoricals, strings or the index, Report_Ref
    data = get_value_cols_only(df)

    # ensure indexes are available, will be important later
    assert data.index.name == 'Report_Ref', "Report_Ref must be index returned by get_value_cols_only()"
    assert df.index.name   == 'Report_Ref', "Report_Ref must be index on df submitted to this function"

    # get column headers of value columns
    cols = data.columns

    # get column headers of non-value (ie string/categorical) columns
    cols_categ = [colname for colname in df.columns if colname not in cols]

    # now we can log1p
    translog1p = FunctionTransformer(np.log1p, validate=True)
    data = translog1p.fit_transform(data)

    # standardise (subtract mean, divide by sd)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(data) #OR transformer = preprocessing.RobustScaler().fit(X)
    data   = scaler.transform(data)

    # return data to DataFrame format
    data = pd.DataFrame(data=data, columns=cols, index=df.index)

    # recombine the data with ;
    # reference column, data itself and categorical columns
    # note, none have an index, as that would prevent 
    data_return = pd.concat([data[cols],      # value columns (standardised and logged)
                            df[cols_categ]], # category or string columns, no change
                            axis=1)

    # also return the sd's and means, so scaling can be reversed
    sds   = pd.Series(data=scaler.scale_, index=cols)
    means = pd.Series(data=scaler.mean_,  index=cols)

    return data_return, sds, means

# apply to our data sets
prepped_cols, sds_cols, means_cols = log_standardise(sec_cols)

#save results
prepped_cols.to_csv(os.path.join(proj_root, 'prepped_10cols.csv'), mode='w', header=True, index=True)
sds_cols.to_csv(os.path.join(proj_root, 'sds_10cols.csv'), mode='w', header=True, index=True)
means_cols.to_csv(os.path.join(proj_root, 'means_10cols.csv'), mode='w', header=True, index=True)

#%%
################## TRAIN, TEST & VALIDATION DATA SETS ###########################

# split into test, train and validation
# BY COMPANY NOT BE RECORD!
# We are planning time series analysis
# so would like as many complete time series of company reports as possible
# This means keeping all reports of a given company within one data set; train, test or validation
# So we must select those sets by company, not simply by record

# prepped_10cols = pd.read_csv(os.path.join(proj_root, 'prepped_10cols.csv'),index_col='Report_Ref')

def train_test_split_df(data, propns=[0.8, 0.1, 0.1], seed=20190925, method='by_company'):

    # propns = [train, valid, test]
    # so 0.8, 0.1, 0.1 means 80% training, 10% validation, 10% test
    
    # ensure data is shuffled
    data = data.sample(frac=1)

    if method == 'by_company':
        # get list of unique companies with the data
        # remember company ID is the 'cik' component of the Report_Ref
        # plus the coreg (subsidiary name)
        data_comp = pd.Series(data.index.str.extract(r'(-[0-9]*_[^_]*_)').loc[:,0])

        # get distinct companies+coreg
        comps_dist = data_comp.unique()

        #Some companies report many more times than others
        # data_comp_ct = data_comp.value_counts().reset_index(drop=False)
        # data_comp_ct.columns = ['company', 'count']

        # prepare to randomly sample the lsi tof distinct companies
        np.random.seed(seed)

        # create random sequence same length as distinct companies series
        rand_seqnc = np.random.rand(len(comps_dist))

    else: # mask by record
        # get random sequence as long as data (much longer than distinct companies!)
        np.random.seed(seed)
        rand_seqnc = np.random.rand(len(data))

    train_mask = rand_seqnc < propns[0]
    valid_mask = [True if value < (propns[0] + propns[1]) and value > propns[0] else False for value in rand_seqnc]
    testi_mask = rand_seqnc > (propns[0]+propns[1])

    if method == 'by_company':

        # find the Report_Ref using company name, ie first 20 digits.
        mask_by_company = lambda mask: data.loc[ data.index[ data_comp.isin(comps_dist[mask]) ] ]
        
        # apply mask
        data_train = mask_by_company(train_mask)
        data_valid = mask_by_company(valid_mask)
        data_testi = mask_by_company(testi_mask)
        
    else: # mask by record
        # apply mask
        data_train = data[train_mask]
        data_valid = data[valid_mask]
        data_testi = data[testi_mask]

    return data_train, data_valid, data_testi

#%%
# Apply the above function

# This is an autoencoder, the target y = the input x, no need for separate targets
# get data split for 10col data
# prepped_cols = pd.read_csv(os.path.join(proj_root, 'prepped_10cols.csv'), index_col='Report_Ref')

x_train_cols, x_valid_cols, x_testi_cols = train_test_split_df(data=prepped_cols)

print("Shape of training set: ",  x_train_cols.shape)
print("Shape of validation set: ",x_valid_cols.shape)
print("Shape of testing set: ",   x_testi_cols.shape)

#%% 
# Save data to file

x_train_cols.to_csv(os.path.join(proj_root, 'x_train_10cols.csv'), mode='w', header=True, index=True)
x_valid_cols.to_csv(os.path.join(proj_root, 'x_valid_10cols.csv'), mode='w', header=True, index=True)
x_testi_cols.to_csv(os.path.join(proj_root, 'x_testi_10cols.csv'), mode='w', header=True, index=True)
