#%%
### ENVIRONMENT ###

# SOFTWARE:
#   Windows 10 Pro 64bit
#   Python 3.6.5, Anaconda3
#   Visual Studio Code (VS Code)
#       Presented in Jupyter Notebooks format for VSCode
#       where #%% indicates a code chunk

# HARDWARE: 
#   CPU: Intel core i5 7500 (4 core)
#   GPU: NVIDIA GeForce RTX 2070 (ASUS) 8GB 
#   RAM: 32GB RAM
#   HDD: 512GB SSD (OS), 1TB SSD (Data)

import sys
print(sys.version)

#%%
### EXECUTIVE SUMMARY ###

# 20-Aug-2019

# The Objective is to assist business planning using AI (deep learning) methods
# The output is a generative deep learning tool (Variational Autoencoder)
# whose latent space is a simplified representation of all the company reports
# currently available from the UK's Companies House, which is millions of reports

# The hope is that we can plot a line from one report to another in latent space
# And this represents a reasonable path for developing a business from one
# set of results (perhaps poor or small company) to a larger, more successful one
# Ultimately, this could be developed with probabilities for each path from a to b
# and perhaps even give an indication of the time dimension 
# ie how much time is usually necessary for such a transition

# The applications are for investors/creditors wishing to gauge a business plan
# and for companies themselves, wishing to understand a business plan's credibility
# The deep model acts as an expert who has read millions of company reports
# and developed a statistical intuition about how they develop and interrelate

# The first half of the notebook is concerned with downloading 
# and parsing the data from Companies House.

# The second half is concerned with developing the Variational Autoencoder
# The output is a usable model with recommendations to investigate other models 
# using the same data

# Overall the data is explored using 10 (ten) deep learning architectures. 
# These are developed in subsequent python notebooks (VSCode):

# VAE's (developed in tf.keras on Tensorflow 2 with Eager Execution)
#   VAE developed for single reports and for time series of reports
#   MMD-VAE developed for single reports and for time series of reports
#   Vector Quantised Variational Autoencoder (VQ-VAE)

# GANs (developed in Keras on Tensorflow 1.17)  
#   Generative Adversarial Network (GAN)
#   Wasserstein GAN - to encourage smooth learning. GANs learn erratically..
#   InfoGAN - to enforce a meaningful latent space

# Autoencoder
#   whose reconstruction losses are used to gauge 'report likeness' of the generated data from GAN or VAE

# Autoencoder with Critic
#   a novel architecture designed to improve accuracy of the 'report likeness' of the generated data

# The MMD VAE is found to be the optimal model for this data.
# However, more data is required, a future projct will parse the reports available
# from the SEC in the USA.

#%%
############ PREPARE PANDAS TABLE OF COMPANY REPORT FOLDERS TO BE DOWNLOADED ##################################

# Companieshouse.gov.uk publishes UK company reports
#   http://download.companieshouse.gov.uk/en_monthlyaccountsdata.html

# Similar data is also available from the SEC in the USA
#   https://www.sec.gov/dera/data/financial-statement-data-sets.html
#   Reports submitted to SEC are also in xbrl format in .htm files
#   BUT, the above link is to a text file summary of those xbrl's
#   This needs another parser to be written! FUTURE WORK!

# As regards the company reports from Companies House...
# Most are still manually submitted as PDF, which is not easily computer readable
# Many businesses exist which translate those reports into structured data, but at substantial cost to the subscriber
# Thnakfully, many companies choose to automatically submit to Companies House in .xbrl format
# This is computer readable. Companies House publishes a zipped folder of such reports for each month since 2014
# All we need to do is create a list of those months and the relevant folder name
# in order to download them to our local PC

# TO DOWNLOAD, UNZIP & PROCESS THE XBRL DATA FROM MILLIONS OF REPORTS AVAILABLE FROM COMPANIES HOUSE
# TOOK THIS ENVIRONMENT **6 DAYS** OF PC TIME (4 CORES). Cores are the limiting factor.

from dfply import *
import pandas as pd

df_name = pd.DataFrame({
    'key'   : [1],
    'Name'  : ['Accounts_Monthly_Data-']
    })

df_months = pd.DataFrame({
    'key'     : [1 for x in range(12)],
    'MonthSeq': [x+10 for x in range(12)],
    'Month'   : ['January','February','March','April','May','June',
                 'July','August','September','October','November','December']
    })

df_years = pd.DataFrame({
    'key'   : [1 for x in range(5)],
    'Year'  : [2018, 2017, 2016, 2015, 2014]
    })

df_merged = pd.merge(df_name, df_months, on='key')

df_merged = pd.merge(df_merged, df_years,  on='key')[['Name', 'Month', 'MonthSeq', 'Year']]

df_merged = df_merged >> unite('Sequence', 
                                ['Year', 'MonthSeq'], 
                                sep = '', 
                                remove = False,
                                na_action = 'as_string')

df_merged = df_merged >> unite('filename', 
                                ['Name', 'Month', 'Year'], 
                                sep = '', 
                                remove = False,
                                na_action = 'as_string')

df_merged = df_merged.sort_values(by=['Sequence'], ascending=False).reset_index(drop=True)

print(df_merged)

#%%
############ DOWNLOAD COMPANY REPORT FROM COMPANIES HOUSE ##################################

from tqdm import tqdm
import colorama #needed by tqdm on windows box
import os
import zipfile
import requests 

# set local directory
# Find the data manually at http://download.companieshouse.gov.uk/en_monthlyaccountsdata.html

proj_root = 'E:\Data\CompaniesHouse'

os.chdir(proj_root)

for index, row in df_merged.iterrows():
    filename = row['filename']
    
    if int(float(row['Sequence'])) >= 201810: #post Dec 2017
        url = 'http://download.companieshouse.gov.uk/'+filename+'.zip'
    else:
        url = 'http://download.companieshouse.gov.uk/archive/'+filename+'.zip'
    
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
    

    ##below lines give alternative implementation with progress bar, but also slow!
    #with open(filename, "wb") as handle:
        #for data in tqdm(response.iter_content()):
        #   handle.write(data)

#%%
############ FUNCTIONS TO ASSESS COMPLETED DOWNLOADS ##################################
#Get list of all folders which have been downloaded

import os
import pandas as pd
from os import listdir, chdir
from os.path import isfile, isdir, join
from dfply import *

proj_root = 'E:\Data\CompaniesHouse'

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

#%%
############ LIST COMPLETED DOWNLOAD FOLDERS ##################################

folders_list_df = get_list_of_folders(proj_root)
print(folders_list_df)

#%%
############ FUNCTIONS FOR FOR FILE PARSING ##################################

# Helper functions
# in the data extraction we will need a function to identify which sign (+ or -) values are
import re as re
from pathlib import Path

def apply_value_scale_sign(value, scale, sign):

    # ensure types are as expected
    value = str(value)
    scale = str(scale).strip()
    sign  = str(sign).strip()

    # extract only digits and decimal points from anything in the value field
    regexlist = re.findall(r'(\d+\.\d+)|(\d+)', value) # returns tuple of the | statement

    # if blank then assume 0
    if len(regexlist)==0 :
        regexlist.append(('0',''))

    # concat tuples in each list item, then join all list items to one string
    value = ''.join([str(i)+str(j) for i, j in regexlist])

    # convert to float
    value = float(value)

    # apply sign
    if sign =='-':
        value = value*-1

    # apply scale
    if scale != 'None' and scale != '0' :
        value = value * np.power(10, float(scale))

    return value

#%%
#occassionally the first item in the contents is returned as 'html'
#rather than <ix:nonfraction .....></ix_nonfraction>
#this causes the data extraction code to fail
#so we remove the offending member of the list
def remove_prefix_tag(thesoup):
    if len(thesoup.contents) <= 1:
        progress = 'data missing'
    elif thesoup.contents[0] == 'html':
        thesoup.contents = thesoup.contents[1:]
        progress = 'data extracted'
    else:
        progress = 'data extracted'
        pass #no action, data is good to go
    return thesoup, progress

#%%
#  We run individual tag finds in try catch blocks

def try_except(item, name, *exceptions):
    try:
        return item.get(name)
    except exceptions or Exception:
        return '+parseerror+'

#%%
# we'll need to save data to file
# first write should be new file, subsequent should be append
# here's a function to do that for pandas dataframes
def dataframe_to_file(outputfile_path_and_name, dataframe, file_save_counter):
    # save data to file
    if not (Path(outputfile_path_and_name).is_file()) or (file_save_counter == 0):
        # create new file with headers
        dataframe.to_csv(outputfile_path_and_name, mode='w', header=True, index=False)
    else:
        # append to file without headers
        dataframe.to_csv(outputfile_path_and_name, mode='a', header=False, index=False) 

#%%
# and now, the code which does the hard work of parsing the files
# for an intro to inline xbrl (aka xbrli) see:
# http://www.xbrl.org.uk/techguidance/samples.html
# and
# http://www.xbrl.org/WGN/inlineXBRL-part0/inlineXBRL-part0-WGN-2015-12-09.html

from bs4 import BeautifulSoup, SoupStrainer

def extract_xbrl(filepath, filename):

    #filename = 'Prod224_0058_SO305325_20171231.html'
    #filepath = os.path.join(proj_root, folder)
    try:
        file = open(os.path.join(filepath, filename), "r")
        html = file.read()
        file.close()
        # print('Opened ', filename)
    except:
        print('Error opening ', filename)
        return 'error on file open'

    # test code
    # global filename_test 
    # filename = filename_test

    # get company number and submission date from file name
    crn        = filename[13:21]
    report_date= filename[22:30]

    # The data extraction takes place in a try except block
    # There are millions of files and we cannot anticipate all failure types
    # So if there is a failure we simply return an error "parse error"

    try:

        # filter the document to only those tags with data, labelled 'ix:nonfraction' or 'ix:resources'
        # We cannot do this soup.find_all because the colon (:) in the tag name breaks the find_all method.
        # But soupstrainer seems to work...
        strainer_resources_nonfraction = SoupStrainer(["ix:nonfraction", "ix:resources"])
        soup_resources_nonfraction = BeautifulSoup(html, 'lxml', parse_only=strainer_resources_nonfraction)

        # XBRL is not standardised, believe it or not! I was shocked, I mean, after all that effort to define it, :(
        # Some software creates XBRL fields with 'xbrli:' or 'xbrldi:' prefixed to each tagname
        # we remove this stuff.
        html_resources_nonfraction = str(soup_resources_nonfraction).replace('xbrli:','').replace('xbrldi:','')
        soup_resources_nonfraction = BeautifulSoup(html_resources_nonfraction, 'lxml')

        # we would like to then do soup_resources_nonfraction.find_all('ix:nonfraction').get_text()
        # but the colon (:) in the tag name breaks the find_all method.
        # So we use SoupStrainer to further filter soup_resources_nonfraction down to only "ix:nonfraction" tags
        strainer_nonfraction = SoupStrainer("ix:nonfraction")
        soup_nonfraction = BeautifulSoup(html_resources_nonfraction, 'lxml', parse_only=strainer_nonfraction)

        #occassionally the first item in the contents is returned as 'html'
        #rather than <ix:nonfraction .....></ix_nonfraction>
        #this causes the data extraction code to fail
        #so we remove the offending member of the list
        def remove_prefix_tag(thesoup):
            if len(thesoup.contents) <= 1:
                progress = 'data missing'
            elif thesoup.contents[0] == 'html':
                thesoup.contents = thesoup.contents[1:]
                progress = 'data extracted'
            else:
                progress = 'data extracted'
                pass #no action, data is good to go
            return thesoup, progress

        soup_resources_nonfraction, progress = remove_prefix_tag(soup_resources_nonfraction)  
        soup_nonfraction, progress = remove_prefix_tag(soup_nonfraction)

        # if there is no data, then return None
        # example of report with no data = Prod224_0058_SO305568_20171231.html
        if len(soup_nonfraction) < 1 or len(soup_resources_nonfraction) < 1:
            return None

        # extract all data from document as one list, multiple data types per list
        # doing it this way means we make only one pass of the data
        # accelerating the process with functional programming. 'For loop' much slower

        f1 = lambda tag: tag.get('name')
        f2 = lambda tag: tag.get('contextref')
        f3 = lambda tag: tag.get('unitref')
        f4 = lambda tag: tag.get('scale')
        f5 = lambda tag: try_except(tag, 'sign') # we defined this 'try_except' function above
        f6 = lambda tag: tag.get_text()

        # we need a nested for loop, each function to be passed over each tag
        # this is how we do a nested for loop in functional programming, using list comprehension:
        rowrec = [function(tag) for tag in soup_nonfraction.contents for function in (f1,f2,f3,f4,f5,f6)]

        # reshape the list into a list of lists
        items_per_row = 6
        rowrec_as_listoflists = [rowrec[i:i + items_per_row] for i in range(0, len(rowrec), items_per_row)]

        # change format to pandas
        rowrec_as_df = pd.DataFrame(rowrec_as_listoflists, 
            columns=['item', 'contextref', 'units', 'scale', 'sign', 'value_nosign'])
        rowrec_as_df['crn'] = crn
        rowrec_as_df['report_date'] = report_date

        # extract true value from 'value', 'sign' and scale
        rowrec_as_df['value'] = np.vectorize(apply_value_scale_sign, otypes=[float])(
                                                rowrec_as_df['value_nosign'], 
                                                rowrec_as_df['scale'],
                                                rowrec_as_df['sign'])
        # separate out the item field into item and reporting standard (eg UK-GAAP) using colon separator
        new = rowrec_as_df['item'].str.split(':', n = 1, expand = True) 
        rowrec_as_df['standard'] = new[0]
        rowrec_as_df['item'] = new[1]

        # In order to extract the "ix:resources" data, we need to know which resources to look for
        # Those are the unit and context resources named in the ix:nonfractional fields
        # Let's get the relevant units first
        units_list = pd.DataFrame(data=rowrec_as_df['units'].unique(), columns=['units'])
        # some files have leading/lagging space which breaks the code
        units_list['units'] = units_list['units'].str.strip()
        # now extract the unit information from the units section of the document
        units_list['units_measure'] = [soup_resources_nonfraction.find(id=unitref).find('measure').get_text() for unitref in list(units_list['units'])]

        # ... and the relevant contexts
        context_list = pd.DataFrame(data=rowrec_as_df['contextref'].unique(), columns=['contextref'])
        
        # some files have leading/lagging space which breaks the code
        context_list['contextref'] = context_list['contextref'].str.strip()

        # now extract the context information from the units section of the document
        # There are two types of context information we want:
        # a) the period (date of the data)
        context_list['period_measure'] = [soup_resources_nonfraction.find(id=context).find('period').get_text() for context in list(context_list['contextref'])]
        # b) the context reference should the data appear in a matrix (aka table)
        nonehandler = lambda member: 'None' if member is None else member.get_text()
        context_list['context'] = [nonehandler(soup_resources_nonfraction.find(id=context).find('explicitmember')) for context in list(context_list['contextref'])]

        # the context ref is prefixed with some text and a colon, we don't need this prefix
        new = context_list['context'].str.split(':', n = 1, expand = True) 
        # must handle situation if there was no colon, hence no split
        if len(new.columns) > 1:
            # colons present, splitting was required for at least some contextrefs
            nonehandler = lambda member: member[0] if member[1] is None else member[1]
            context_list['context'] = [nonehandler(row) for row in zip(new[0], new[1])]

        # where the period is a date range the code returns '\n' within the text, which we need to strip
        context_list['period_measure'] = [measure.replace('\n', '').replace('-', '') for measure in context_list['period_measure']]
        # Something similar can also apply to table contexts, but we retain a space where we had \n
        context_list['context'] = [context.replace('\n', ' ') for context in context_list['context']]

        # if we are not given period start then period_measure reads 'enddate'
        # if we are given period start then it reads: startdateenddate
        # meaning we have to be careful extracting the end date, which is always provided, because it moves from first 8chars to last 8
        # so we flip period_measure around to read enddatestartdate
        context_list['period_measure'] = context_list['period_measure'].astype('str').str[8:] + context_list['period_measure'].astype('str').str[0:8] 

        # now we are safe to extract period_open and period_close from period_measure
        context_list['period_open'] = context_list['period_measure'].astype('str').str[8:]
        context_list['period_close'] = context_list['period_measure'].astype('str').str[0:8]

        #  superfluous columns
        context_list = context_list.drop(columns=['period_measure'])
        rowrec_as_df = rowrec_as_df.drop(columns=['sign', 'value_nosign', 'scale'])

        # join the units and contexts to the main data
        rowrec_as_df = rowrec_as_df >> left_join(units_list, by='units')
        rowrec_as_df = rowrec_as_df >> left_join(context_list, by='contextref')

        # flag companies reporting P&L
        if len(rowrec_as_df[(rowrec_as_df['item'] == 'ProfitLossOnOrdinaryActivitiesBeforeTax')
                            |
                            (rowrec_as_df['item'] == 'OperatingProfitLoss')
                            ]) > 0 :
            rowrec_as_df['ReportIncludesPL'] = 1
        else:
            rowrec_as_df['ReportIncludesPL'] = 0

        # if there is any error in the above parsing then we simply return a string 'parse error'
    except :
        print('Error parsing ', filename)
        return 'parse error'

    return rowrec_as_df

#%%
############ EXECUTE PARSING USING AS MANY CORES AS POSSIBLE ##################################

# loop around the folders, extracting data from each file in each folder
# there will be three output files
# 1. a file of all data found within a folder, which is a backup to the above because its so large!
# 2. a file of all data from all folders, the master record
# 3. a file of all data from all folders which is for companies which publish profit/loss, which is the data we are immediately interested in

# This code processes financial accounts at bewteen 15 to 40 files per second on this PC
# This PC has 2x1TB Samsung Pro SSD, Corei5-7500 (only 4cores) and 32GB HyperX 2133Mhz DDR4 RAM
# There are approx 200,000 xbrl files submitted per month to HMRC
# There are approx 60 months of xbrl data available
# So this code will take 200,000 * 60 / 40 seconds
# Which is 3.5 DAYS!
# Approx 1.5hrs per month (ie folder) of data

# You NEED as many cores as you can get, preferably Xeon with 24cores+, or at least i7 with 8cores

import time
import datetime
from joblib import delayed, Parallel
from tqdm import tqdm

output_file_master = os.path.join(proj_root, 'All_Cos.csv')
output_file_PLonly = os.path.join(proj_root, 'All_CosWithPL.csv')

#flag whether we want to display prgress bars, slows things down in Jupyter
use_tqdm = False

#we need a flag to hlep us decide when to write or append results to file
#start with 0 for initial write (aka 'w' mode which implies overwrite)
file_save_counter = 0

# loop around folders
for f in range(len(folders_list_df.index)):

    # get current folder name
    folder = folders_list_df.loc[f,'foldername']

    # create filename for all results of folder
    output_file_folder = os.path.join(proj_root, folder+'.csv')

    # log commencement
    folders_list_df.loc[f, 'progress'] = 'Commenced'
    print("Just opened this folder: ", folder)

    # set current dir to the chosen folder
    os.chdir(os.path.join(proj_root, folder))

    # this takes a long time, so let's use a timestamp to find out how long
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(folder," started at: ", st)

    # get list of files in folder
    files_list_df = get_list_files_in_folder(folder)

    # loop around files in folder
    # there are approx 200,000 files in a folder, we need parallel processing to do this in a reaosnable time frame
    # here we use the joblib library to achieve the parallelism
    # we can choose to use tqdm or not, slows things down in Jupyter
    
    if use_tqdm:
        folder_results = Parallel(n_jobs=4)(delayed(extract_xbrl)(os.path.join(proj_root, folder), filename) for filename in tqdm(files_list_df['filename']))
    else:
        folder_results = Parallel(n_jobs=4)(delayed(extract_xbrl)(os.path.join(proj_root, folder), filename) for filename in files_list_df['filename'])
    
    # note there is an alternative package commonly used for parallel processing
    # but it simply hung on this windows pc
    #   import multiprocessing as multip
    #   poolv = multip.Pool(multip.cpu_count())
    #   filtered_xbrl = []
    #   filtered_xbrl = poolv.starmap(extract_xbrl, [(proj_root, filename) for filename in files_list_df['filename']])
    #   poolv.close()

    # remove null reports and reports which errored upon opening, this is a 1GB table, so performance is important
    # note, filter() is 50% faster than a list comprhension, which was tried like this:
    #   outcomes_jl = [dataframe for dataframe in outcomes_jl if type(dataframe) == pd.DataFrame]
    folder_results = filter(lambda dataframe: type(dataframe) == pd.DataFrame, folder_results)

    # concatenate the records for this folder into one large pandas dataframe
    # because currently stored as a list of dataframes
    folder_results_pd = pd.concat(folder_results)

    # massive file, so free up space immediately
    del folder_results

    # save data
    # 1. to file for folder's data only
    dataframe_to_file(output_file_folder, folder_results_pd, file_save_counter=file_save_counter)
    # 2. to master file of all folder's data
    dataframe_to_file(output_file_master, folder_results_pd, file_save_counter=file_save_counter)
    # 3. to file for companies reporting PL
    dataframe_to_file(output_file_PLonly, folder_results_pd[folder_results_pd['ReportIncludesPL'] == 1], file_save_counter=0)

    # increment file_save_counter so subsequent file saves are appends
    file_save_counter += 1

    # log completion of folder
    folders_list_df.loc[f, 'progress'] = 'Completed'

    #print finish time
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(folder," finished at: ", st)


#%%
############ INSPECT PARSED DATA ##################################

#Let's take a quick peak at the data to confirm we can at least read the data for our own company
# The main file (all companies) is too big to load into memory. 
# The file is 25GB, will be larger when loaded. This server has 32Gb and will not load the entire file
# So we use dask
# http://docs.dask.org/en/latest/dataframe.html

import dask.dataframe as dd
All_Cos = dd.read_csv(os.path.join(proj_root, 'All_Cos.csv'), dtype={'date_valid': 'str','crn': 'str', 'report_date':'str'})

# How many records in the table?
print("Records in the All_Cos table: ", len(All_Cos))

#%%
# can we find our own company data?
All_Cos_self = All_Cos[All_Cos.crn == '07362441'].compute()

# note because the above command finished .compute(), the dask file is calculated and a pandas object created
print(All_Cos_self.head())


#%%
#How many currencies and units are we dealing with?
print(All_Cos.units.unique().compute())


#%%
# initially we'll analyse companies which report P&L, because its much more informative than BS
# the file of companies reporting P&L is only 1GB, manageable in memory
# so don't need dask, use pandas directly
All_CosWithPL = pd.read_csv(os.path.join(proj_root, 'All_CosWithPL_20190531.csv'), low_memory=False)

# how many companies report a P&L over the period?
# select unique companies
DistinctCos = All_CosWithPL >> distinct(X.crn) >> select(X.crn)
DistinctRep = All_CosWithPL >> \
                select(X.item, X.crn, X.period_close) >> \
                    filter_by(X.item == 'ProfitLossOnOrdinaryActivitiesBeforeTax') >> \
                        group_by(X.crn, X.period_close) >> \
                            summarize(count=X.item.count())

print("Number of companies reporting a P&L in the period: ", len(DistinctCos.index))
print("Number of P&L reports in the period: ", len(DistinctRep.index)) # every report listed twice, current and previous

#%%
# we'll use ggplot for plotting (not matplotlib)
# http://ggplot.yhathq.com/
# https://github.com/yhat/ggpy

# You may suffer from 'pandas has no attribute tslib'
# https://github.com/yhat/ggpy/issues/662
# so to get ggplot to import properly, you may need:
# a) get your site library location:
#   from distutils.sysconfig import get_python_lib
#   print(get_python_lib())
# b) make change to ggplot package which needs update
# see issue at: https://stackoverflow.com/questions/31003994/anaconda-site-packages
# Open file ../site-packages/ggplot/stats/smoothers.py
# Change from pandas.lib import Timestamp to from pandas import
# Timestamp in line 4
# Change pd.tslib.Timestamp to pd.Timestamp in line 14.
# Save the file

from ggplot import *
# Let's look at the distribution of profits
ProfitDistrib = All_CosWithPL >> \
                    select(X.item, X.value, X.crn, X.period_close) >> \
                        filter_by(X.item == 'ProfitLossOnOrdinaryActivitiesBeforeTax') >> \
                            group_by(X.crn, X.period_close) >> \
                                summarize(ProfitLossB4Tax = X.value.mean())

# print(ProfitDistrib.ProfitLossB4Tax.max())
# print(ProfitDistrib.ProfitLossB4Tax.min())

p = ggplot(ProfitDistrib, aes('ProfitLossB4Tax')) + \
    geom_histogram(color='black', binwidth=100000) + \
    xlim(-1000000,1000000)

print(p)


#%%
# How many reports with > 100k profit (substantial business)

ProfitCountOver100k = All_CosWithPL >> \
                        select(X.item, X.value, X.crn, X.period_close) >> \
                            filter_by(X.item == 'Equity', X.value > 100000 )
                       
print("How many reports show equity > 100k? (ie substantial business): ",ProfitCountOver100k.value.count())

#%%
# What is the distribution of data over time

ProfitDistribMonth = All_CosWithPL >> \
                        mutate(yrmth_at_close=X.period_close.astype('str').str[0:6]) >> \
                            filter_by(X.item == 'ProfitLossOnOrdinaryActivitiesBeforeTax') >> \
                                select(X.crn, X.yrmth_at_close)

q = ggplot(aes(x='yrmth_at_close'), data=ProfitDistribMonth) + \
    geom_bar() + \
    ggtitle('Reports Submitted Per Mth') + \
    xlab('reports') + \
    ylab('month') #+ \
    #theme(axis.text.x = element_text(angle=90,hjust=1,vjust=0.5))
print(q)

#%%
# How many employees do these companies have?

Employees = All_CosWithPL >> \
                select(X.item, X.value, X.crn, X.period_close) >> \
                    filter_by(X.item == 'AverageNumberEmployeesDuringPeriod') >> \
                        group_by(X.crn, X.period_close) >> \
                            summarize(EmployeeCount = X.value.mean())

r = ggplot(Employees, aes('EmployeeCount')) + \
    geom_histogram(color='black', binwidth=5) +\
    xlim(0,1000)

print(r)

#%%

# There are a handful of companies with -ve quantity of employees, 
# upon investigation this turned out to be a parsing error. Quantities are correct, but sign is +ve
# Let's correct that

def update_employees(item, value):
    if item == 'AverageNumberEmployeesDuringPeriod' and value < 0 :
        value = -1 * value
    return value

All_CosWithPL['value'] = np.vectorize(update_employees, otypes=[float])(All_CosWithPL['item'], All_CosWithPL['value'])

#%%
################## MAP ITEMS TO SIMPLE GROUPS ###########################

# company accounts contain line items which make distinctions that do not interest this research
# for example, there are many different income types, but our initial model only wants to know
# about operating income or finance income
# For another example, admin expenses are often reported in many different flavours, that can usefully be grouped 'Admin expenses'
# So, we need a mapping table which maps the item into a simplified group and the report to which it belongs, being PL, BS, Equity or Cash Flow
# at its most basic we only need the SoI (aka P&L), SOFP (aka Balance Sheet) and some notes, such as qty of employees
# This simplification will also help the model learn as many items are used in a tiny minority of reports, but can be fairly mapped to a simpler group of items

# Let's find what the most common reporting items and an associated example

# most common items...
All_CosWithPL_items = All_CosWithPL >> select(X.item, X.crn, X.report_date)

Items = All_CosWithPL_items >> \
            select(X.item) >> \
                group_by(X.item) >> \
                    summarize(ItemCount = X.item.count())

# its' useful to find an example of each...
Item_Examples = (All_CosWithPL_items.assign(
                            sequence=All_CosWithPL_items.sort_values(['crn', 'report_date'], ascending=True)
                                .groupby(['item'])
                                .cumcount() + 1
                                )
                        .sort_values(['item','crn','report_date']))

Item_Examples = Item_Examples >> filter_by(X.sequence == 1)

# bring these queries together...
Items = Items >> inner_join(Item_Examples, by='item')

# In this project flow the above table of items was exported to a CSV file and manually mapped to simpler groups of items
# eg Creditors_Trade and Creditors are both mapped to simple 'Creditors'
Items.sort_values(by='ItemCount', ascending=False).to_csv('E:\\Data\\Correction_Test\\items.csv', mode='w', header=True, index=False)


#%%
################## DOWNLOAD NAMESPACES AND CREATE MAPPING TABLES ###########################
# Each item belongs to a namespace, to understand which report, (SOI, SOFP, selected notes)
# # an item belongs to we need namespace taxonomies. These report names (SOFP etc) are NOT given in the XBRL itself, very annoyingly

# http://www.xbrl.org.uk/techguidance/taxonomies.html

# On links found in the above URL there are mapping tables in Excel. These show which item belongs to which report, be it PL, BS, Cash Flow etc
# These excel mapping tables need to be amended to the mapping suitable for our usage, just the right amount of detail
# This needs to be done manually and cross referenced with the most frequently cited items in our own data (see above chunk)
# between these two tables a hand crafted selection of 

# now upload the table of mappings and apply to our data
MappingTable_Core = pd.read_csv(os.path.join(proj_root, 'MappingTable_Core.csv'), low_memory=False)

# How many items are mapped as being 'core' to understanding a company?
# This will also indicate the width of the table, ie the number of dimension, presented to the model
print("Number of items used to summarise a company's position = ", len(MappingTable_Core['item_mapped'].unique()))

#%%
###################################################################################################
# Prepping data for the model
###################################################################################################

# Some items are listed more than once because they appear in a table both as an opening and a closing figure
# Such a table may be presented for both the current year and previous year, 
# meaning an item may be presented 4 times
# The next year's report would then repeat the figures from the previous year
# In other words, this data is full of dupes
# We need to discard duplicates and label the sequence (open, close)

# The first step is to filter down to the items used in the mappings, greatly reduces the data to be manipulated
# let's constrain the data to the mapped items and rename them as the 'items to be considered.
All_CosWithPL_Prepped = All_CosWithPL >> \
                            inner_join(MappingTable_Core, by='item') >> \
                                select(~X.item) >> \
                                    rename(item = X.item_mapped) >> \
                                        select(X.crn, X.item, X.period_open, X.period_close, X.value, X.report_date)

# we can easily discard duplicates within the groupings of crn, report_date, item, period_close
All_CosWithPL_Prepped.drop_duplicates(subset=['crn', 'report_date', 'item', 'period_close'], keep='first', inplace=True)

# Now, a single report may still have an item listed more than once, one mention for each period_close
# We need a common way of listing these, preferably in reverse chronological order
# in SQL we would use ROW_NUMBER() OVER(PARTITION BY crn, item ORDER BY period_close) 
# to give a reverse chronological sequence per item

# first get list of unique dates per report
All_CosWithPL_Prepped_daterng = All_CosWithPL_Prepped >> \
                                    select('crn', 'report_date', 'period_close') >> \
                                        distinct('crn', 'report_date', 'period_close')

# Here's how to do that window aggregate in pandas...
All_CosWithPL_Prepped_daterng = (All_CosWithPL_Prepped_daterng.assign(
                                    sequence = All_CosWithPL_Prepped_daterng.sort_values(['period_close'], ascending=False)
                                        .groupby(['crn', 'report_date'])
                                        .cumcount() + 1
                                    )
                                .sort_values(['crn','report_date','sequence']))

# only current and previous period_close are permitted, ie sequence=1 and sequence=2
All_CosWithPL_Prepped = All_CosWithPL_Prepped >> \
                            inner_join(All_CosWithPL_Prepped_daterng, by=['crn','report_date','period_close']) >> \
                                filter_by(X.sequence <= 2) 

# tag the item name with whether it is an current or previous figure
# this will be important when we pivot the data into columns as dimensions for the model
All_CosWithPL_Prepped['item'] = All_CosWithPL_Prepped['item'] + ['_curr' if sequence == 1 else '_prev' for sequence in All_CosWithPL_Prepped['sequence']]

# clean up
del All_CosWithPL_Prepped_daterng

#%%

# There will be only one row of data per report. 
# How to represent the date of that data?
# We will use period_close and the number of months in the period (may be less than 12)
# we will not use report_date, which is based purely on when the reprot was submitted to companies house
# Here's how to extract a single period_close and period_duration figure from each report
# In SQL it would be MAX(period_close) OVER(PARTITION BY crn, report_date) 
All_CosWithPL_MaxClose = All_CosWithPL_Prepped >> \
                            select (X.crn, X.report_date, X.period_close) >> \
                                group_by(X.crn, X.report_date) >> \
                                    summarize(MaxClose=X.period_close.max())

All_CosWithPL_MaxOpen  = All_CosWithPL_Prepped >> \
                            select (X.crn, X.report_date, X.period_open) >> \
                                group_by(X.crn, X.report_date) >> \
                                    summarize(MaxOpen=X.period_open.max())

All_CosWithPL_Prepped  = All_CosWithPL_Prepped >> \
                            inner_join(All_CosWithPL_MaxClose, by=['crn', 'report_date']) >> \
                            inner_join(All_CosWithPL_MaxOpen,  by=['crn', 'report_date'])  

del All_CosWithPL_MaxClose, All_CosWithPL_MaxOpen

# convert to datetime format
All_CosWithPL_Prepped['MaxOpen']  = pd.to_datetime(All_CosWithPL_Prepped['MaxOpen'],  format='%Y%m%d', errors='coerce')
All_CosWithPL_Prepped['MaxClose'] = pd.to_datetime(All_CosWithPL_Prepped['MaxClose'], format='%Y%m%d', errors='coerce')

# get duration
All_CosWithPL_Prepped['Duration'] = All_CosWithPL_Prepped['MaxClose'] - All_CosWithPL_Prepped['MaxOpen']

# Return the date types to float for the sake of the model
# Both as complete months since Jan 1970
All_CosWithPL_Prepped['MaxClose'] = round(pd.to_timedelta(All_CosWithPL_Prepped['MaxClose']) / np.timedelta64(1,'M'))
All_CosWithPL_Prepped['Duration'] = round(All_CosWithPL_Prepped['Duration'] / np.timedelta64(1,'M'))

# create new field as concatenate of crn and report_date
# this is a reference to the report and will not be submitted to the model
All_CosWithPL_Prepped['Report_Ref'] = All_CosWithPL_Prepped['crn'] + '_' + All_CosWithPL_Prepped['report_date'].astype(str)

#  superfluous columns
All_CosWithPL_Prepped = All_CosWithPL_Prepped.drop(columns=['sequence', 'crn', 'report_date', 'period_close', 'period_open', 'MaxOpen'])

# save results
All_CosWithPL_Prepped.to_csv(os.path.join(proj_root, 'All_CosWithPL_Prepped.csv'), mode='w', header=True, index=False)

#%%
################## PIVOT THE DATA ###########################

# We now pivot the data such that each row is a company report and each column is a key item in that report (Sales, Expenses, Profit, etc)
All_CosWithPL_Prepped_pvt = All_CosWithPL_Prepped.pivot(index='Report_Ref', columns='item', values='value')

# The above pivot moved the Report_Ref column into the index, we can reverse this using reset_index()
All_CosWithPL_Prepped_pvt.reset_index()

# The pivot process also removed the Duration and MaxClose fields, but is easy to replace them
DurationAndMaxClose = All_CosWithPL_Prepped >> \
                        select(X.Report_Ref, X.MaxClose, X.Duration) >> \
                            distinct(X.Report_Ref, X.MaxClose, X.Duration)

All_CosWithPL_Prepped_pvt = All_CosWithPL_Prepped_pvt >> \
                                inner_join(DurationAndMaxClose, by='Report_Ref')

# re-arrange column order, Duration and MaxClose have been joined to the end of a very wide table
# its more convenient if they are at the left of the table.
cols = All_CosWithPL_Prepped_pvt.columns.tolist()
cols = ['Report_Ref','MaxClose','Duration', *cols[1:len(cols)-2]]
All_CosWithPL_Prepped_pvt = All_CosWithPL_Prepped_pvt[cols]

# save results
All_CosWithPL_Prepped_pvt.to_csv(os.path.join(proj_root, 'All_CosWithPL_Prepped_pvt.csv'), mode='w', header=True, index=False)

#%%
################## INSPECT THE DATA ###########################
# A lot of work has gone into reducing the 750GB of company reports into a concise 'face' for each company report which includes profit/loss
# Worth one last look before proceeding to construction of the model

# how sparse is our matrix? How many rows (ie company reports) have <X% of the data completed
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

Completeness_row, Completeness_col = get_completeness(All_CosWithPL_Prepped_pvt)

#%%
################## REDUCE SPARSITY ###########################

# So, the most complete rows are only 45% complete
# This is a very sparse matrix
# The first step is to remove rows and columns the vast majority of which are blank
# Any deep learning system will likely guess '0' for such columns all the time, and be untrainable

# First let's remove COLUMNS of data which are hardly ever completed, we want > 10,000 entries out of the 136,000 possible
Selected_cols = Completeness_col >> filter_by(X.cells_completed > 10000) >> arrange(X.cells_completed, ascending=False)
All_CosWithPL_Prepped_pvt_notsparse = All_CosWithPL_Prepped_pvt[['Report_Ref','MaxClose','Duration'] + Selected_cols['colname'].tolist()]
Completeness_row, Completeness_col = get_completeness(All_CosWithPL_Prepped_pvt_notsparse)

# We are left with 108 columns to define our company reports, the face of the company, just a 10x10 image!

# Now let's remove ROWS of data which are mostly blank entries, we want > 30 entries out of the 105 possible
Selected_rows = Completeness_row >> filter_by(X.cells_completed > 30) >> arrange(X.cells_completed, ascending=False)
All_CosWithPL_Prepped_pvt_notsparse = All_CosWithPL_Prepped_pvt_notsparse.iloc[Selected_rows['rownumber']]
Completeness_row, Completeness_col = get_completeness(All_CosWithPL_Prepped_pvt_notsparse)

#summarise the situation
print("\n","Data dimensions are now: ", All_CosWithPL_Prepped_pvt_notsparse.shape)

# save results
All_CosWithPL_Prepped_pvt_notsparse.to_csv(os.path.join(proj_root, 'All_CosWithPL_Prepped_pvt_notsparse.csv'), mode='w', header=True, index=False)

#%%
################## MANAGE REMAINING SPARSITY ###########################

# We now have a better populated table with >80,000 rows, enough for deep learning
# But, there are still many missing records and we cannot submit missing records to a deep learning system, we must decide on how such records are to be represented, if at all

## Option 1: Replace missing data with 0
#   REJECTED: this will not be investigated further. 0 has a very specific meaning for all the acctg headings
#   It would be deeply misleading to suggest turnover or assets are 0, simply because data is missing.

## Option 2: Select rows and records such that there are no missing data points
#   Let's iterate over the columns and and see how many complete rows we get for each quantity of columns
#   The data has already been organised such that most complete columns are at left

column_list = list(All_CosWithPL_Prepped_pvt_notsparse.columns[3:])
results = pd.DataFrame(columns=['Column_Qty', 'Complete_Rows', 'Column_List'])
results['Column_List'] = results['Column_List'].astype(object)

for attempt in range(1,50):
    completes_prev = 0
    completes = 0

    for i in range(1,len(column_list)):

        completes_prev = completes
        columns = [column_list[k] for k in list(range(0,i))]

        # how many rows have no missing data?
        completes = sum(All_CosWithPL_Prepped_pvt_notsparse[columns].notna().sum(axis=1) == i)

        # save results
        if attempt == 49:
            results.loc[i-1,'Column_Qty'] = i
            results.loc[i-1,'Complete_Rows'] = completes
            results.loc[i-1,'Column_List'] = column_list

        if i == 1:
            completes_prev = completes

        # if we suddenly lost 25% of the rows or more, then we have a counter indicator column, remove column from list of columns to review
        # a counter indicator is a field which is common for small companies, but uncommon for big companies
        # meaning it has many companies which report it, but those companies report few other things.
        # its the opposite of what we need
        # Havin removed this column form the list, we need to make another attempt (loop) through the remaining columns
        if completes/completes_prev < 0.75:
            column_list.remove(column_list[i-1])
            break

print(results)

#%%
# The result of the above loop is that we can have 
# a) 50,000 company reports if we settle for only 10 columns
print("The ten columns completed for 50,000 company reports are:")
cols10 = pd.Series(column_list[0:10]).sort_values()
print(cols10)

All_CosWithPL_Prepped_pvt_notsparse_10col = All_CosWithPL_Prepped_pvt_notsparse[['Report_Ref','MaxClose','Duration'] + list(cols10)]
completes = All_CosWithPL_Prepped_pvt_notsparse_10col.notna().sum(axis=1) == (10+3) # add the three ref columns
All_CosWithPL_Prepped_pvt_notsparse_10col = All_CosWithPL_Prepped_pvt_notsparse_10col.iloc[list(completes)]
print("\n")

# b) 30,000 company reports if we settle for only 20 columns
print("The twenty columns completed for 30,000 company reports are:")
cols_20 = pd.Series(column_list[0:20]).sort_values()
print(cols_20)

All_CosWithPL_Prepped_pvt_notsparse_20col = All_CosWithPL_Prepped_pvt_notsparse[['Report_Ref','MaxClose','Duration'] + list(cols_20)]
completes = All_CosWithPL_Prepped_pvt_notsparse_20col.notna().sum(axis=1) == (20+3) # add the three ref columns
All_CosWithPL_Prepped_pvt_notsparse_20col = All_CosWithPL_Prepped_pvt_notsparse_20col.iloc[list(completes)]

# save results
All_CosWithPL_Prepped_pvt_notsparse_10col.to_csv(os.path.join(proj_root, 'All_CosWithPL_Prepped_pvt_notsparse_10col.csv'), mode='w', header=True, index=False)
All_CosWithPL_Prepped_pvt_notsparse_20col.to_csv(os.path.join(proj_root, 'All_CosWithPL_Prepped_pvt_notsparse_20col.csv'), mode='w', header=True, index=False)

#%%

# Nw we have our candidate pandas tables let's engage in a little more data exploration
# This uses the fantastic pandas_profiling package, whose use I'd like to document here...

import pandas_profiling

All_CosWithPL_Prepped_pvt_notsparse_10col = pd.read_csv(os.path.join(proj_root, 'All_CosWithPL_Prepped_pvt_notsparse_10col.csv'))
All_CosWithPL_Prepped_pvt_notsparse_20col = pd.read_csv(os.path.join(proj_root, 'All_CosWithPL_Prepped_pvt_notsparse_10col.csv'))

# prepare report on 10cols data
pandas_profiling.ProfileReport(All_CosWithPL_Prepped_pvt_notsparse_10col)

# prepare report on 20cols data
pandas_profiling.ProfileReport(All_CosWithPL_Prepped_pvt_notsparse_20col)

# Result is that lots of the features are skewed
# We'll probably need to log the data
# Also we see that the Profit before tax is highly correlated with profit after tax, which is obvious really
# It was left this way because effective tax rate can have na impac ton a business, we really wanted the tax paid, the difference between these two figures.

#%% [markdown]
# These maybe just about feasible but we lose a great deal of interest and most of our example company reports

## Option 3: Get clever with out
#   Deep learning models are routinely asked to learn from subsets of the data, perhaps we can use that to our advantage
#   papers have been published on this
#   REJECTED: Time is too short to investigate this specialist avenue

## Option 4: Impute values
#   REJECTED: From person experience imputing values for companies works when we have say 80% of the data available, otherwise it can be veyr misleading
#   Interestingly, Autoencoder De-noisers impute values by learning from complete sets of data first
#   But our data is skewed such that only large companies publish complete data sets
#   Would it be reasonble to train on big companies (ie complete data) then attempt to impute missing values of small companies?
#   In effect, this option is the ame as option 1, since we first train on complete data sets

## Option 5: Present the model with ALL data AND a flag for each cell indicating whether the cell is NA
#   There we have twice as many cells as before, 
#       effectivley one table of data and another table with categoricals 1 or 0 indicating whether the cell contains data
#   This would allow the model to learn about the reporting requirements of small businesses
#   Whilst we retain all our examples, hence a wider sample of businesses
#   Missing data would be set to 0

################## OPTION 5 SELECTED - SEE NEXT CHUNK ###########################

#%% [markdown]
################## MANAGE NEGATIVE VALUES ###########################

# Before we implement option 5 (representing missing values with categorical column)
# We have one more problem to resovle. Spoiler - it has the same solution...

# Most likely we will end up log transforming the data before it is submitted to the model
# this is because money is power distributed, you have to move though the lower
# values of wealth to reach the higher ones, like poisson distribution. Such is life.
# So we have a skew to low values, already very visible in the charts above

# Such distributions stop neural nets from learning, 
# so we log them to make then as 'normal' as possible with a simple, reversible, function.
# BUT, we cannot log -ve values

## Option 1: abs(value)
#   REJECTED: We cannot we simply provide the model with +vs absolute values, that greatly distorts information
#   A loss would appear to be the same as profit.

## Option 2: Translate up to zero
#   REJECTED: It is not reasonable to simply translate the data by the minimum value (add the mins, so that min presented to model is zero)
#   The 'minimums' are different for each data set, the translations would be different for train vs validation vs test and any future data set
#   Within this data a human could not be sure of what was a loss and what was a profit, a model could not learn either
#   It might be possible to select some outragously high figure to be added to each value, so large that we'd be certain of never 
#   having a -ve figure any greater magnitude. But as the data is logged, the signal would be lost to this enormous bias.

## Option 3:
#   Supply the absolute (+ve) value of each item AND a category to indicate that it is -ve or +ve
#   Where we already have categories, eg indicating missing values, we simply add a new category

# Easy to create a 'shadow' table of flags and append to the right of the data
# first sort columns

#%%

def get_value_cols_only(df):
    '''
        Removes non values columns, ie categoricals and strings, from dataframe
        Uses col names to achieve this, must be disciplined about column names!
        Cannot use dtypes because categoricals are 1 or 0, ie dtype=int64, 
        TAKES: dataframe
        RETURNS: dataframe
    '''
    # we can only transform columns which are values, but not string and not categorical
    # remove the report_ref column (string)
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
    df_to_map = df.drop(columns=['Report_Ref','MaxClose','Duration']) 

    # ignore any categoricals already existing
    df_to_map = get_value_cols_only(df_to_map)

    # save those categoricals for later
    df_categs = df.drop(columns=[*df_to_map.columns.to_list(), *['Report_Ref','MaxClose','Duration']])

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
    return pd.concat([df[['Report_Ref','MaxClose','Duration']], df_to_map, df_categs, new_cols], axis=1)

# apply pos/neg function to the two simple options of data representation which EXCLUDE NA's
data_10cols = apply_cat(All_CosWithPL_Prepped_pvt_notsparse_10col, posneg, '_ispos')
data_20cols = apply_cat(All_CosWithPL_Prepped_pvt_notsparse_20col, posneg, '_ispos')

# apply pos/neg AND isna function to option 5 data, which INCLUDE NA's
data_catego = apply_cat(All_CosWithPL_Prepped_pvt_notsparse, posneg, '_ispos')
data_catego = apply_cat(data_catego, notna, '_notna')

# save results
data_catego.to_csv(os.path.join(proj_root, 'data_catego.csv'), mode='w', header=True, index=False)
data_10cols.to_csv(os.path.join(proj_root, 'data_10cols.csv'), mode='w', header=True, index=False)
data_20cols.to_csv(os.path.join(proj_root, 'data_20cols.csv'), mode='w', header=True, index=False)


#%%
################## GET DISTRIBUTIONS OF DATA ###########################

# We know at least some of the data is log distributed, employee numbers for example.
# This would be a problem for the deep learning model as this kind of data slows or even prevents learning
# We need to know what the distribution of each field is, then pre-process accordingly
# Courtesy of stack overlfow: 
# https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3
# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python?lq=1

import scipy.stats as st
def get_best_distribution(series, bins=200):
    '''
        Gets the most likely distirbution of a series by comparing with standard distirbutions
        available in scipy.stats
        TAKES: series (not dataframe)
        RETURNS: tuple of best fitting distribution name and p value
    '''
    dist_names = ["norm",     # centralised
                  "powerlaw", # skewed to higher numbers
                  "expon",    # skewed to lower numbers
                  "uniform"]  # flat
                  # there are lots of others of course, 
                  # https://docs.scipy.org/doc/scipy/reference/generated
    dist_results = []
    params = {}

    for dist_name in dist_names:

        # get the candidate distribution
        dist = getattr(st, dist_name)

        # fit it to the series
        param = dist.fit(series)

        # get the params of the fit
        params[dist_name] = param

        # Applying the Kolmogorov-Smirnov test to get quality of fit
        D, p = st.kstest(series, dist_name, args=param)

        # Append the quality of fit to list of candidates for the series
        dist_results.append((dist_name, p))

    # select the best fitting candidate to the series
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))

    return best_dist, best_p
#%%
# Let's apply the tool and see if it makes sense. Each column shold have 10,000 points to plot

def get_distrib_per_col(df):
    '''
        Applies the function 'get_best_distribution' across each column in a dataframe
        TAKES: dataframe of data to analyse
        RETURNS: dataframe summarising most likely distirbution for each column
    '''
    # prep dataframe to hold results
    item_distrib = pd.DataFrame(columns=['Item', 'Distrib', 'Pval'], dtype=str)

    # separate out the values from the categoricals (whose distribution we ar enot interested in)
    data_no_categ_cols = get_value_cols_only(df)

    # loop over value columns
    for col_name in data_no_categ_cols.columns:

        # ensure blanks are removed
        data_series  = data_no_categ_cols[col_name]
        data_series  = data_series[data_series.notnull()]

        # get best distribution
        best_dist, best_p = get_best_distribution(data_series)

        # save result for the column
        row_to_add   = pd.DataFrame(data={'Item': str(col_name), 'Distrib': best_dist, 'Pval':best_p}, index=[0])
        item_distrib = item_distrib.append(row_to_add)

    return(item_distrib)

# Test it on the 10col data (simplest data set)
item_distrib = get_distrib_per_col(data_10cols)

print(item_distrib)

# This doesn't work well, it suggestes normal for all columns, with pval= 0.0

#%%
# We suspected exponentional distribution, it said normal
# Let's double check that we do have exponential...
u = ggplot(data_10cols, aes('001_SOI_TurnoverRevenue_curr')) + \
    geom_histogram(color='black', binwidth=100000) #+ \
    #xlim(-1000000,1000000)

print(u)

# As we suspected, VERY exponential!

#%%
################## LOG AND SCALE ###########################
# The above code is typically not very successful
# From inspection we know most are exponentially distributed (preference for low numbers)
# This is because money is poisson distributed, it is a 'count',
# where we have to pass through low numbers to get to high numbers, such is the path to creating wealth!
# So, we will log and then standardise ALL features

# https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9

#get pipeline processing tools
# https://scikit-learn.org/stable/modules/preprocessing.html
from sklearn.preprocessing import FunctionTransformer, StandardScaler

def log_standardise(df):
    '''
        Logs and Standardises (subtract by mean, div by sd) value data (not categoricals)
        Intended for feeding into neural net
        TAKES: dataframe
        RETURNS: dataframe
    '''

    # ensure we log and scale only value fields, not categoricals or strings
    data = get_value_cols_only(df)

    # get column headers
    cols = data.columns

    # now we can log1p
    translog1p = FunctionTransformer(np.log1p, validate=True)
    data = translog1p.fit_transform(data)

    # standardise (subtract mean, divide by sd)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(data) #OR transformer = preprocessing.RobustScaler().fit(X)
    data = scaler.transform(data)

    # return data to DataFrame format
    data = pd.DataFrame(data=data, columns=cols)

    # recombine the data with reference column, data itself and categorical columns
    data_return = pd.concat([df['Report_Ref'].reset_index(drop=True), 
                             data.reset_index(drop=True), 
                             df.drop(columns=[*cols, 'Report_Ref']).reset_index(drop=True)], 
                             axis=1)

    # set report_ref as the index, so not fed into models but available for reference
    data_return = data_return.set_index('Report_Ref')
    sds   = pd.Series(data=scaler.scale_, index=cols)
    means = pd.Series(data=scaler.mean_, index=cols)

    # we now have values to reverse the scaling and log process
    # scaler.scale_, scaler.mean_, 
    return data_return, sds, means

# apply to our data sets
prepped_10cols, sds_10cols, means_10cols = log_standardise(data_10cols)
prepped_20cols, sds_20cols, means_20cols = log_standardise(data_20cols)
prepped_catego, sds_catego, means_catego = log_standardise(data_catego)

#save results
prepped_10cols.to_csv(os.path.join(proj_root, 'prepped_10cols.csv'), mode='w', header=True, index=True)
sds_10cols.to_csv(os.path.join(proj_root, 'sds_10cols.csv'), mode='w', header=True, index=True)
means_10cols.to_csv(os.path.join(proj_root, 'means_10cols.csv'), mode='w', header=True, index=True)

prepped_20cols.to_csv(os.path.join(proj_root, 'prepped_20cols.csv'), mode='w', header=True, index=True)
sds_20cols.to_csv(os.path.join(proj_root, 'sds_20cols.csv'), mode='w', header=True, index=True)
means_20cols.to_csv(os.path.join(proj_root, 'means_20cols.csv'), mode='w', header=True, index=True)

prepped_catego.to_csv(os.path.join(proj_root, 'prepped_catego.csv'), mode='w', header=True, index=True)
sds_catego.to_csv(os.path.join(proj_root, 'sds_catego.csv'), mode='w', header=True, index=True)
means_catego.to_csv(os.path.join(proj_root, 'means_catego.csv'), mode='w', header=True, index=True)

#%%
################## TRAIN, TEST & VALIDATION DATA SETS ###########################

# split into test and train

def train_test_split_df(data, propns=[0.7, 0.1, 0.2], seed=20190613):

    # propns = [train, valid, test]
    # so 0.7, 0.1, 0.2 means 70% training, 10% validation, 20% test

    # get random sequence as long as data
    np.random.seed(seed)
    rand_seqnc = np.random.rand(len(data))

    # training
    train_mask = rand_seqnc < propns[0]

    # validation
    valid_mask = [True if value < (propns[0] + propns[1]) and value > propns[0] else False for value in rand_seqnc]

    # testing
    testi_mask = rand_seqnc > (propns[0]+propns[1])

    # return data
    data_train = data[train_mask]
    data_valid = data[valid_mask]
    data_testi = data[testi_mask]

    return data_train, data_valid, data_testi

# This is an autoencoder, the target y = the input x, no need for separate targets
# get data split for 10col data
x_train_10cols, x_valid_10cols, x_testi_10cols = train_test_split_df(data=prepped_10cols)

print("Shape of 10col training set: ",x_train_10cols.shape)
print("Shape of 10col validation set: ",x_valid_10cols.shape)
print("Shape of 10col testing set: ",x_testi_10cols.shape)

x_train_10cols.to_csv(os.path.join(proj_root, 'x_train_10cols.csv'), mode='w', header=True, index=True)
x_valid_10cols.to_csv(os.path.join(proj_root, 'x_valid_10cols.csv'), mode='w', header=True, index=True)
x_testi_10cols.to_csv(os.path.join(proj_root, 'x_testi_10cols.csv'), mode='w', header=True, index=True)

# get data split for categorical data (largest dataset)
x_train_catego, x_valid_catego, x_testi_catego = train_test_split_df(data=prepped_catego, propns=[0.8, 0.1, 0.1])

print("Shape of 10col training set: ",   x_train_catego.shape)
print("Shape of 10col validation set: ", x_valid_catego.shape)
print("Shape of 10col testing set: ",    x_testi_catego.shape)

x_train_catego.to_csv(os.path.join(proj_root, 'x_train_catego.csv'), mode='w', header=True, index=True)
x_valid_catego.to_csv(os.path.join(proj_root, 'x_valid_catego.csv'), mode='w', header=True, index=True)
x_testi_catego.to_csv(os.path.join(proj_root, 'x_testi_catego.csv'), mode='w', header=True, index=True)

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
################## TEST FUNCTION TO UN LOG AND UN SCALE ###########################

# let's see if our function works...

unlogscale_10cols = un_log_scale(df = prepped_10cols, 
                                 df_sd = sds_10cols, 
                                 df_mn = means_10cols, 
                                 reinstate_neg = False, 
                                 reinstate_nan = False)

# compare records
data_10cols_test = data_10cols.set_index('Report_Ref')
errors = unlogscale_10cols - data_10cols_test
errors_col = errors.max(axis=0)

# print results
print("Maximum error in the re-consituted data = ", max(errors_col))

# This appears to work fine, max error is 1x10-6 (a millionth of a pound).

#%%
###################################################################################################
# Building a Variational Autoencoder
###################################################################################################

# We'll use a variational auto encoder to tell us about the 'latent space' which companies inhabit
# by companies we really mean company reports, latent space is the minimal representation of such a report, its 'essence'
# the variational autoencoder will allow us to view a 'path' (ie vector) between companies
# showing the possible/plausible stages between one type of business and another
# which, of course, is the heart of business planning; 

# Introduction to autoencoders at http://kvfrans.com/variational-autoencoders-explained/
# Code inspired by example at https://github.com/keras-team/keras/blob/keras-2/examples/variational_autoencoder.py
# Advanced details inc min batch size limit of 100 at https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/

# FUTURE WORK
# Will develop MMD-VAE and VQ-VAE, as inspired by these RStudio blogs... (in R)
#   https://blogs.rstudio.com/tensorflow/posts/2018-10-22-mmd-vae/
#   https://blogs.rstudio.com/tensorflow/posts/2019-01-24-vq-vae/

#%%

from keras.layers import Input, Dense, Lambda, LeakyReLU, BatchNormalization, concatenate
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras import metrics
from keras import backend as K
from keras.utils import plot_model

# confirm we are running on GPU, not CPU
from tensorflow.python.client import device_lib
print([(item.name, item.physical_device_desc) for item in device_lib.list_local_devices()])

#%%
################## SAMPLING FUNCTION ###########################
# 
# the heart of the variational autoencoder is sampling of the latent space
# We need this function before we can build the models.

def sampling(args):
    '''
        Samples the latent space by monte carlo sampling from the standard deviation
        and adding the mean. In other words: z = z_mean + K.exp(0.5 * z_log_var) * epsilon
        TAKES: z_mean, z_log_var
        RETURNS: z (tensor): sampled latent vector
    '''
    # extract variables from args. Both are tensors of shape: (batch, latent_space_nodes)
    z_mean, z_log_var = args

    # we want to monte carlo sample the latent space using a perturbation, epsilon
    # the standard deviation of this perturbation is one, mean is 0
    epsilon_std = 1.0
    epsilon_mn  = 0.0

    # The random numbers tensor needs shape (batch_size, latent_space_nodes)
    batch_dim   = K.shape(z_mean)[0]
    latent_dims = K.int_shape(z_mean)[1]

    # create a tensor of epsilon (aka perturbation or monte carlo) values 
    # by default we assume the distribution of those random values has mean=0, sd=1
    # random tensor shape is (batch_size, latent_space_nodes) but not all batches same size (final batche is leftovers, so dont define)
    epsilon = K.random_normal(shape = (batch_dim, latent_dims), 
                              mean  = epsilon_mn,
                              stddev= epsilon_std)

    # z, the sample from the distribution, is simply mean + a multiple of variance.
    # The multiple is epsilon, which is itself normally distributed
    # result is that we are taking a random normal sample from the distribution of z
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    return z

#%% [markdown]
################## CUSTOM LOSS FUNCTION ###########################

# As per the link (above) to kvfrans blog...
# there's a tradeoff between how accurate our network can be and 
# how close its latent variables can match the unit gaussian distribution.
# We want to minimise both, so for our loss term we sum the two separate losses: 
#   the generative loss:
#       which is a mean squared error that measures how accurately the network reconstructed the record
#   the latent loss:
#       which is the KL divergence that measures how closely the latent variables match a unit gaussian
# IN SUMMARY:
#   reconstruction_loss = mean(square(generated_record - real_record))
#   latent_loss         = KL Divergence(latent_variable, unit_gaussian)
#   combined_loss       = reconstruction_loss + latent_loss

#Note, the 'latent loss' is the Kullback-Leibler divergence between a prior distribution
# imposed on the latent space (typically, a standard normal distribution) and the 
# representation of latent space as learned from the data. 
# Minimising this loss is intended to incentivse the model to force the data in the latent space
# into a normal distribution.

# we can alternatively use binary_crossentropy instead of mse
# or even have two losses, bxent for binary fields and mse for regression fields
# This is has yet to be implemented, simple first...

#%%

# Note how this custom loss function returns a function, 
# which is where the nested function uses its access to the arguments of the enclosing function...aka 'closure'
# Defining it this way allows us to provide the loss function with more than the standard
# loss function variables of inputs and outputs
# see https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618

# NB LOSS FUNCTIONS MUST GIVE SMOOTH CHANGES IN LOSS OVER TIME (ie between batches)
# ie MUST BE DIFFERENTIABLE
# SO AVOID MIN(), MAX(), ROUND() etc, AS THESE ARE 'NON HOMOGENOUS', ie THEY PREVENT SMOOTH CHANGES
# IF A LOSS FUNCTION GIVES THE SAME LOSS OVER MANY EXAMPLES THEN THE MODEL WEIGHTS WILL CEASE TRAINING

def model_loss(z_log_var, z_mean, original_dim, has_nan, recon_loss_method):

    def vae_loss(inputs, outputs):

        # this is a combination of regression fields (use mse?) and binary category field (use binary cross entropy?)
        # if the data is not sparse (does not have nan's), then we have 2+n regression fields + n categoricals which denote whether the field is negative or not.
        # The first two fields are Duration and Date. They have no categoricals associated with them
        
        # There are three ways the reconstruction loss could be calculated. These ar eoptions at compile time
        # for two of the methods we need to know which columns are regressional and which categorical

        # This depends on whether there are fields suffixed 'isnan'
        if not has_nan :
            # First we split out input and output tensors into the regression and categorical fields...
            colqty = (original_dim - 2)/2
        else :
            colqty = (original_dim - 2)/3

        # tensor indices must be integers
        colqty = int(colqty)

        # now we can proceed with the reconstruction loss functions
        
        if recon_loss_method == 'simple_mse':
            # the simplest reconstruction loss is good old fashioned MSE (typically used in regressionals)
            # without ANY acknowledgement that some fields are categorical and others regressional
            reconstruction_loss = metrics.mse(inputs, outputs)

        elif recon_loss_method == 'simple_bxent':
            # the next simplest reconstruction loss is binary cross entropy (typically used in categoricals)
            # without ANY acknowledgement that some fields are regressional and others categorical
            reconstruction_loss = metrics.binary_crossentropy(inputs, outputs)

        elif recon_loss_method == 'medium':
            # the next reconstruction loss is MSE for regression fields and binary cross entropy for categorical
            # this acknowledges that some columns are categorical and others regressional
            # we then just add the two errors, meaning it remains differentiable
            
            # loss in regression fields
            inputs_regression  = K.slice(inputs, start=[0,0], size=[-1,colqty+2])
            outputs_regression = K.slice(outputs,start=[0,0], size=[-1,colqty+2])
            
            reconstruction_loss_reg = metrics.mse(inputs_regression, outputs_regression)

            # loss in binary (1 or 0) category fields
            inputs_categ  = K.slice(inputs, start=[0,colqty+2], size=[-1,colqty])
            outputs_categ = K.slice(outputs,start=[0,colqty+2], size=[-1,colqty])

            reconstruction_loss_categ = metrics.binary_crossentropy(inputs_categ, outputs_categ)

            # Stack the two tensors of losses. 
            # Each tensor is dim (batch), but needs to be (batch,2) in order to apply mean()
            stack = K.stack([reconstruction_loss_reg, reconstruction_loss_categ], axis=1)

            # mean of the two losses
            reconstruction_loss = K.mean(stack, axis=1)

        elif recon_loss_method == 'complex':
            # the most complex method attempts to increase regression error where there is a category error. 
            # Eg if regression field is accurate, but of the wrong sign (+ve, rather than -ve)
            # then the regression error is actually twice as big.

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
      
        # Now we have our reconstruction loss, we must attend to the latent loss
        # latent_loss = KL-Divergence(latent_variable, unit_gaussian)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        # get combined loss
        combined_loss  = reconstruction_loss + kl_loss
        
        return combined_loss

    # For loss calculation keras expects a function to be returned
    return vae_loss

#%%
################## FUNCTION TO CREATE VAE ###########################

# VAE model = encoder + decoder
# The encoder is simply a compression algorithm which has been trained to best fit the data
# we'll write functions to create the model components; encode+throat, decode
# using a function allows us to grid search hyperparams (eg qty of layer nodes) later
# Activation functions are important, we use leaky_relu, which can be forced into being simple relu
#   https://isaacchanghau.github.io/post/activation_functions/

def define_and_compile_vae( input_df,
                            encode_layer_nodes,
                            decode_layer_nodes,
                            reconstruction_cols,
                            recon_loss_method,
                            loss_function       = model_loss,
                            has_nan             = False,
                            latent_space_nodes  = 2,
                            sampling_function   = sampling,
                            apply_mc_sampling   = True,
                            apply_batchnorm     = False,
                            leaky_alpha         = 0.0,
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

    # Latent Space & its Sampling
    ####################################################################
    # This is the key aspect of Variational Autoencoders; sampling the latent space
    # We create two parameters from latent space. Assumed to be: z_mean and z_log_sigma. 
    # Then, we randomly sample some points 'z' from the latent normal distribution,
    # which is simply assumed to exist.
    # Those points are defined by z = z_mean + sqrt(var) * epsilon
    # where epsilon is a random normal tensor (a perturbation term forcing random sampling)
    z_mean    = Dense(latent_space_nodes, name='z_mean')(encoded)
    z_log_var = Dense(latent_space_nodes, name='z_log_var')(encoded)

    # Note the above two layers are in parallel and intentionally HAVE NO activation
    # In effect the sampling function is the activation
    # get those samples, output shape will be (latent_space_nodes, batch_size)
    # we can switch off monte-carlo sampling and return only the mean with 'epsilon_is_zero = True'
    z = Lambda(sampling_function, output_shape=(latent_space_nodes,), name='z')([z_mean, z_log_var])
    
    # having defined layers, create graph
    #   inputs = input batch
    #   output = a list of the z tensors for the batch: z_mean, z_log_var and z
    encoder = Model(inputs=inputs_encoder, outputs=[z_mean, z_log_var, z], name='encoder')

    ####################################################################
    # Decoder
    ####################################################################

    # Input to Decoder (from latent space)
    inputs_decoder = Input(shape=(latent_space_nodes,), name='z_sampling')

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
    # VAE = Encoder + Decoder
    ####################################################################

    # ENCODE DATA INTO LATENT SPACE
    z_mean, z_log_var, z = encoder(inputs_encoder)

    # DECODE FROM LATEBT SPACE & RECONSTRUCT
    outputs_decoder = decoder(z)

    # VAE = Encoder + Decoder
    vae = Model(inputs = inputs_encoder, outputs = outputs_decoder, name = 'vae')

    # Compile VAE, using loss defined above. Must pass closure variables to function
    vae.compile(optimizer = 'adadelta', loss = loss_function(z_log_var, z_mean, original_dim, has_nan, recon_loss_method))

    return encoder, decoder, vae

#%%
################## FUNCTION TO SAVE KERAS MODELS ###########################

from contextlib import redirect_stdout

def save_model(model_object, filename, history, save_history=False, proj_root=proj_root):
    
    # save keras model (ie serialize and send to file) 
    model_object.save(os.path.join(proj_root,'SaveModels',filename+'_model.h5'))

    # save weights only (as backup)
    model_object.save_weights(os.path.join(proj_root,'SaveModels',filename+'_weights.h5'))

    # save summary text
    filename_txt = os.path.join(proj_root,'SaveModels',filename+'_summary.txt')
    with open(filename_txt, 'w') as f:
        with redirect_stdout(f):
            model_object.summary()
    
    # save graph image
    filename_png = os.path.join(proj_root,'SaveModels',filename+'_graph.png')
    plot_model(model_object, to_file=os.path.join(proj_root,'SaveModels',filename_png), show_shapes=True)
   
    # save training history
    #if save_history:
    filename_history = os.path.join(proj_root,'SaveModels',filename+'_history.npy')

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
        TAKES: the input dataframe
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
                    filename = ('data_'    + str(data_name) + 
                                '_lossfn_'+ str(recon_loss_method) + 
                                '_alpha_' + str(leaky_alpha) +
                                '_norm_'  + str(apply_batchnorm) +
                                '_batch_' + str(batch_size))

                    # create model
                    enc, dec, vae = define_and_compile_vae(
                                        input_df           = x_train, 
                                        encode_layer_nodes = encode_layer_nodes, 
                                        decode_layer_nodes = decode_layer_nodes,
                                        reconstruction_cols= get_col_spec(x_train),
                                        latent_space_nodes = latent_space_nodes, 
                                        sampling_function  = sampling,
                                        recon_loss_method  = recon_loss_method,
                                        loss_function      = model_loss,
                                        has_nan            = has_nan,
                                        apply_batchnorm    = apply_batchnorm, 
                                        leaky_alpha        = leaky_alpha,
                                        kinit              = 'glorot_normal')
                    # fit model
                    history = vae.fit(  
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
                    models = [(enc, filename+'_enc'), (dec, filename+'_dec'), (vae, filename+'_vae')]

                    for each_model in models:
                        model_object, model_name = each_model
                        #only save history for vae, not enc and not dec
                        if model_name[0:3] == 'vae':
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
            
    # save to file because this took a long time (approx 10hrs) to complete!
    results.to_csv(os.path.join(proj_root,'SaveModels','GridScan_Results_'+data_name+'.csv'))

#%%
################## GRID SEARCH OF VAE MODELS ###########################

#####################
## For 10cols data ##
params =  { 
           # these params will be looped around in the grid
           'recon_loss_method' : ['medium', 'simple_bxent', 'simple_mse', 'complex'],
           'leaky_alpha'       : [0.4, 0.2, 0.0],
           'apply_batchnorm'   : [False],
           'batch_size'        : [100],   # batch must be >=100 for vae, due use of monte carlo for loss (see link to tiao.io)
           
           # these params will NOT be looped around, mostly lists of ONE object
           'data_name'         : ['10col'],
           'has_nan'           : [False],
           'encode_layer_nodes': [16, 8,  4],
           'decode_layer_nodes': [4,  8, 16],
           'latent_space_nodes': [2],     # code cannot yet handle 3 or more latent dims. So choose 2!
           'earlystop_patience': [20],    # epochs to run before giving up on seeing better validation performance 
           'epochs'            : [200]
           } 
           
# do grid search, NB no return value. Function saves all results to file
grid_search(x_train = x_train_10cols, 
            x_valid = x_valid_10cols,
            params  = params)

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



#%%
# VIEW RESULTS
# which model was best for each data set?

results_10col = pd.read_csv(os.path.join(proj_root,'SaveModels','GridScan_Results_10col.csv'), index_col=0)

best_models_10col = results_10col >> \
                        group_by(X.data_set) >> \
                            summarize(best_valid = X.best_valid.min()) >> \
                                inner_join(results_10col, by=['data_set','best_valid'])

print(best_models)

#%%
################## FUNCTION TO PLOT TRAINING HISTORY OF ONE MODEL ################

def plot_training_history(model_history_filepath):
    model_history = np.load(model_history_filepath)
    model_history = pd.DataFrame(data= model_history, columns=['training_loss', 'validation_loss'])
    model_history['Epoch'] = model_history.index

    plot = ggplot(model_history, aes(x='Epoch')) + \
            geom_line(aes(y = 'training_loss', colour = 'green')) + \
            geom_line(aes(y = 'validation_loss', colour = 'grey')) + \
            labs(x="Latent dim x", y="Latent dim y", title="Latent Space of Company Reports. Colour=Turnover")

    return plot

#%%
################## FUNCTION TO PLOT EMBEDDING (LATENT SPACE) OF ONE MODEL #######

def plot_embeddings(encoder, df):
    # get encodings (ie latent space, not reconstructions) for test data
    x_testi_encoded = encoder.predict(df)

    # outputs from encoder = [z_mean, z_log_var, z]
    # so to see examples read the FIRST item in the list; z_mean
    # we don't want z (third item) because it has a random perturbation, we just want the mean
    x_testi_z = x_testi_encoded[0]

    # convert test embeddings to pandas
    x_testi_z = pd.DataFrame(data=x_testi_z, columns=['x','y'])

    #let's see if we can get a colour from the Turnover...
    x_testi_z['turnover_curr'] = df['001_SOI_TurnoverRevenue_curr'].reset_index(drop=True)

    # plot
    plot = ggplot(x_testi_z, aes(x='x', y='y', color='turnover_curr')) + \
            geom_point() + \
            scale_color_gradient(low='blue', high='red') + \
            labs(x="Latent dim x", y="Latent dim y", title="Latent Space of Company Reports. Colour=Turnover")

    return plot

#%%
############ FUNCTION TO PLOT ALL TRAINING HISTORIES OR ALL EMBEDDINGS ##################################

import matplotlib.pyplot as plt
from keras.models import load_model

def matrix_of_plots(data_name_5char,      # '10col' or 'catego'
                    test_data_df=None,    # test data in pandas
                    plot_type='histories' # 'embeddings' or 'histories'
                    ):

    # get all files
    save_models    = get_list_files_in_folder(os.path.join(proj_root, 'SaveModels' ))['filename']

    # filter down to training histories or embeddings
    if plot_type == 'embeddings':
        save_models = [file for file in save_models if file[-12:]=='enc_model.h5' and file[0:10] == 'data_' + data_name_5char]
        prefix = 'Embeddings for '
        xlabel = 'latent x'
        ylabel = 'latent y'
    else: # training history
        save_models = [file for file in save_models if file[-15:]=='vae_history.npy' and file[0:10] == 'data_' + data_name_5char]
        prefix = 'Training histories for '
        xlabel = 'epochs'
        ylabel = 'loss'

    # how many rows of training figures?
    rows = len(save_models) // 3
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
        loss_fun    = file_name[lossfn_locn+7:alpha_locn-1]
        alpha       = file_name[alpha_locn+6:alpha_locn+9]
        title       = 'Ls:'+loss_fun+', Al:'+alpha

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
            encoder = load_model(os.path.join(proj_root,'SaveModels',file_name), compile=False)

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
            tng_history = np.load(os.path.join(proj_root,'SaveModels',file_name))

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
    plt.savefig(os.path.join(proj_root, 'SaveModels', prefix + data_name_5char + ' data.jpg'),  dpi=300)

#%%
################## FUNCTION TO CALCULATE RECONSTRUCTION ERRORS #########################

from sklearn.metrics import log_loss

def get_reconstruction_losses(inputs, outputs, has_nan, recon_loss_method='complex'):
    # basically the same approach as the reconstruction loss in the custom loss function (see above)
    # except using numpy, as opposed to Keras backend
    # see the above custom loss function for notes
    original_dim = inputs.shape[1]

    if not has_nan :
        colqty = (original_dim - 2)/2
    else :
        colqty = (original_dim - 2)/3

    colqty = int(colqty)

    if recon_loss_method == 'medium':

        # loss in regression fields
        inputs_regression   = inputs[:,0:2+colqty]
        outputs_regression  =outputs[:,0:2+colqty]

        reconstruction_loss_reg = np.mean(np.abs(inputs_regression, outputs_regression))

        # loss in binary (1 or 0) category fields
        inputs_categ  = inputs[:,2+colqty:]
        outputs_categ =outputs[:,2+colqty:]*0.75+0.001

        reconstruction_loss_categ = log_loss(inputs_categ, outputs_categ)

        reconstruction_losses = np.mean(reconstruction_loss_reg, reconstruction_loss_categ)
    
    else:
        reconstruction_abs_errors_A = np.abs(inputs[:,0:2] - outputs[:,0:2])

        inputs_regression   = inputs[:,2:2+colqty]
        outputs_regression  =outputs[:,2:2+colqty]
        reconstruction_abs_errors_B = np.abs(inputs_regression - outputs_regression)

        inputs_categ_ispos  = inputs[:,2+colqty:2+2*colqty]
        outputs_categ_ispos =outputs[:,2+colqty:2+2*colqty]

        ispos_abs_errors = np.abs(outputs_categ_ispos - inputs_categ_ispos)
        ispos_abs_errors = ispos_abs_errors + 1

        reconstruction_abs_errors_B = reconstruction_abs_errors_B * ispos_abs_errors
        concat = np.concatenate((reconstruction_abs_errors_A , reconstruction_abs_errors_B), axis=1)

        reconstruction_losses = np.mean(concat, axis=1)

    return reconstruction_losses

#%%
############ FUNCTION TO INSPECT GRID OF RANGE OF COMPANY REPORTS ##################################

# Our first example of being 'generative'
# build a report generator that samples a grid of generated companies
# by decoding points from latent space
# Those points are selected to represent a normal distribution in the x and y latent spaces

def get_grid_of_samples(df, df_sd, df_mn, decoder, reinstate_neg, reinstate_nan):
    '''
        Builds matrix of latent values and generates company reprot for each one
        TAKES: a df of test data (for its columns), data means, data sd's, a decoder
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
            recon_rpt = decoder.predict(z_sample)

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
############ FUNCTION TO INSPECT STEPS BETWEEN EXAMPLE COMPANY REPORTS #############################

# Get location of two companies in latent space and then sample steps between those two points
# These steps could be considered the 'business plan' to change from one business to another

def get_steps_btwn(encoder, decoder, report_a, report_b, df_sd, df_mn, steps=10, reinstate_neg=True, reinstate_nan=False):
    '''
        Gives steps (ie business plan) between company a and company b
        TAKES: two company reports
        RETURNS: 'step' qty of business reports 
    '''
    # create blank table for results
    colnames  = report_a.columns

    # get embeddings of two company reports. encoder outputs [z_mean, z_log_var, z]. We want z.
    embedding_a = encoder.predict(np.array(report_a))[0]
    embedding_a_pd = pd.DataFrame(data=embedding_a, columns=['latent_x','latent_y'], index=report_a.index)

    embedding_b = encoder.predict(np.array(report_b))[0]
    embedding_b_pd = pd.DataFrame(data=embedding_b, columns=['latent_x','latent_y'], index=report_b.index)

    # create dataframe to hold all z_samples, kick off with sample a
    z_samples = embedding_a_pd

    # create dataframe to hold all reports, kick off with report a
    recon_rpts= report_a

    # To move from a to b we subtract the point a from point b, hence the vector between them
    btwn_vector = embedding_b - embedding_a

    # calculate vector for one step
    btwn_vector_step = btwn_vector / (steps+1)

    # sample 'step' qty of latent points along the vector between a and b
    for step in range(1, steps+1):

        # get next sample and save for future reference
        z_sample = embedding_a + btwn_vector_step * step
        z_samples= z_samples.append(
                    pd.DataFrame(data   = z_sample, 
                                 columns= ['latent_x','latent_y'], 
                                 index  = pd.Index(['step_%03d'%step], name = 'Report_Ref')
                                 ))

        # reconstruct company report from latent space representation
        recon_rpt = decoder.predict(z_sample)  

        # save company report for bulk unscale / unlog later
        recon_rpts= recon_rpts.append(
                    pd.DataFrame(data   = recon_rpt, 
                                 columns= colnames,
                                 index  = pd.Index(['step_%03d'%step], name = 'Report_Ref')
                                 ))

    # set report b as the last item in the list
    recon_rpts = recon_rpts.append(report_b)
    z_samples = z_samples.append(embedding_b_pd)

    # unlog and unscale the data
    recon_rpts_unlog = un_log_scale(df    = recon_rpts,
                                    df_sd = df_sd,
                                    df_mn = df_mn,
                                    reinstate_neg = reinstate_neg,
                                    reinstate_nan = reinstate_nan)

    # return the embeddings (z_samples) along side their unscale/unlogged company report
    recon_rpts_unlog = pd.concat([z_samples, recon_rpts_unlog], axis=1, sort=False)

    return recon_rpts_unlog

#%%
############ INSPECT TRAINING ON 10COLS DATA #####################
matrix_of_plots(data_name_5char='10col', plot_type='histories')

# Result is that only the simple_bxent loss function led to substantial training. 
# Some training with complex loss function.
# Alpha has minor effect on training.
# Alpha = 0.2 seems to works well and trains longest
# Other training failed on validation set

#%%
############ INSPECT EMBEDDINGS ON 10COLS DATA ##################################
matrix_of_plots(data_name_5char='10col', test_data_df = x_testi_10cols, plot_type='embeddings')

# Remember only the simple_bxent loss function led to substantial training
# Some training with complex loss function (alpha=0 and 0.2)
# Alpha has major effect on embeddings
# Loss=Complex, Alpha=0.2 gives a 'gaussian' shaped blob, which is what we want
# Loss=Simple_bxent, Alpha=0.4 gives a well graded embedding across profits, but not 'gaussian'

# 10col CHOSEN: Loss=Complex, Alpha=0.2 / Loss=Simple_bxent, Alpha=0.4

#%%
############ INSPECT TRAINING ON CATEGO DATA #####################
matrix_of_plots(data_name_5char='categ', plot_type='histories')

# Result is that only the complex loss function led to substantial training. 
# Alpha has minor effect. 
# Alpha = 0.2 or 0.4 works best. 0.4 trains longest
# Other training failed on validation set

#%%
############ INSPECT EMBEDDINGS ON CATEGO DATA ##################################
matrix_of_plots(data_name_5char='categ', test_data_df = x_testi_catego, plot_type='embeddings')

# Embeddings for chosen params (Loss=Complex, Alpha=0.4) are a gaussian blob centred at zero. Good!

# CATEGO CHOSEN: Loss=Complex, Alpha=0.2 / Loss=Complex, Alpha=0.4

#%%
################## LOAD CHOSEN 10COLS MODEL & PRE-TRAINED WEIGHTS ###########################

# define selected model file names
file_short = '10col'
file_vae = os.path.join(proj_root, 'SaveModels', 'data_10col_lossfn_simple_bxent_alpha_0.4_norm_False_batch_100_vae_model.h5')
file_enc = os.path.join(proj_root, 'SaveModels', 'data_10col_lossfn_simple_bxent_alpha_0.4_norm_False_batch_100_enc_model.h5')
file_dec = os.path.join(proj_root, 'SaveModels', 'data_10col_lossfn_simple_bxent_alpha_0.4_norm_False_batch_100_dec_model.h5')

# Load the selected models so we can use them for predictions
# Ensure compile=False else it gets in a tizz about the custom loss function
# we could use the custom_objects option, but passing z_log_var, z_mean etc into the loss function is a problem
vae = load_model(file_vae, compile=False)
enc = load_model(file_enc, compile=False)
dec = load_model(file_dec, compile=False)

# select appropriate testing data
x_testi = x_testi_10cols
sds = sds_10cols
mns = means_10cols

#%%
############ INSPECT GRID OF RANGE OF COMPANY REPORTS ##################################
x_testi_grid = get_grid_of_samples(df      = x_testi,
                                   df_sd   = sds,
                                   df_mn   = mns,
                                   decoder = dec,
                                   reinstate_neg = True,
                                   reinstate_nan = False)

x_testi_grid.to_csv(os.path.join(proj_root, 'x_testi_'+file_short+'_grid.csv'), mode='w', header=True, index=False)

#%%
############ INSPECT RECONSTRUCTION LOSSES ##################################

get_reconstruction_losses(np.array(x_testi_10cols[0:10]), 
                          np.array(x_testi_10cols[0:10]),
                          has_nan = False, 
                          recon_loss_method = 'complex')

#%%
############ INSPECT STEPS BETWEEN EXAMPLE COMPANY REPORTS #############################
# get the company reports between which we want to sample steps (points a and b)

# what distance apart should the two reports be?
# should they be for two points which are fairly close, or fairly far apart?
# We'll get Euclidean and cosine distance
# Euclidean maybe misleading in latent space due the number of dims not represented, hence use of cosine 
from scipy.spatial import distance

# first get all points in latent space, ie the embeddings
embeddings = enc.predict(x_testi)[0]

# Euclidean + Cosine distances between all point, returns a pandas df of distances between points
def get_distances_as_list(dist_method='euclidean', remove_dupes=True):
    distances = distance.cdist(embeddings, embeddings, dist_method)

    # option not to repeat distances such that we have 'a to b' and 'b to a'.
    # use np.tril() to zero upper triangle of distances. Later we will filter out the 0 distances
    if remove_dupes:
        distances = np.tril(distances, k=-1) 
        # k indicates which diagonal above which to zero. 
        # k=0 would leave 'a to a' and 'b to b', so we use k=-1

    distances = pd.DataFrame(data = distances, columns = x_testi.index.to_list()) 
    distances['ref'] = x_testi.index
    distances = pd.melt(distances, id_vars=['ref'], value_vars=x_testi.index.to_list())
    distances.columns = ['from','to','distance_'+dist_method]
    
    #now remove distances which have been zeroed (real distances always > zero)
    # this removes 'a to a', 'b to b' etc (distance to self), 
    # it also removes distances which have been zeroed as dupes by np.tril()
    distances = distances[distances['distance_'+dist_method] != 0]
    distances.index = distances['from']+'-'+distances['to']
    
    return distances

# NB We are analysing the embeddings using the test set of 5,000 reports
# Distance between all embeddings requires calculation of the square of 5000 distances
# Which is 25m values. 
# We are NOT using the training set of 50,000, whose square would be 2.5bn distances!

distances_euc = get_distances_as_list(dist_method='euclidean')['distance_euclidean']
distances_cos = get_distances_as_list(dist_method='cosine')['distance_cosine']

# join the two types of distance
distances = pd.concat([distances_euc, distances_cos], axis=1)

# sort by cosine distance
distances = distances.sort_values(by=['distance_cosine'], ascending=True)

# get distribution of distances, mean and variance
distances_stats = distances.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

# get shortlists of distances for both the lowest and highest deciles
lo_quartile_l = distances_stats.loc['min','distance_cosine']
lo_quartile_h = distances_stats.loc['10%','distance_cosine']
hi_quartile_l = distances_stats.loc['90%','distance_cosine']
hi_quartile_h = distances_stats.loc['max','distance_cosine']


#%%

#Get steps between the two reports
# Randomly sleected reports which are close together...

distances_lo_sample = distances[(distances['distance_cosine']>lo_quartile_l) & (distances['distance_cosine']<lo_quartile_h)]
distances_lo_sample = distances_lo_sample.sample(n=1)

from_ref = distances_lo_sample.index[0][0:17]
to_ref   = distances_lo_sample.index[0][18:]

report_a = x_testi.loc[from_ref,:].to_frame().transpose() # taking one row converts to series, but we want dataframe
report_b = x_testi.loc[to_ref,:].to_frame().transpose()

x_testi_samples = get_steps_btwn(encoder  = enc,
                                 decoder  = dec,
                                 report_a = report_a,
                                 report_b = report_b,
                                 df_sd    = sds,
                                 df_mn    = mns,
                                 steps    = 10,
                                 reinstate_neg = True,
                                 reinstate_nan = False
                                 )

x_testi_samples.to_csv(os.path.join(proj_root, 'x_testi_'+file_short+'_samples_close.csv'), mode='w', header=True, index=True)

# Randomly sleected reports which are far apart...
distances_hi_sample = distances[(distances['distance_cosine']>hi_quartile_l) & (distances['distance_cosine']<hi_quartile_h)]
distances_hi_sample = distances_hi_sample.sample(n=1)

from_ref = distances_hi_sample.index[0][0:17]
to_ref   = distances_hi_sample.index[0][18:]

report_a = x_testi.loc[from_ref,:].to_frame().transpose() # taking one row converts to series, but we want dataframe
report_b = x_testi.loc[to_ref,:].to_frame().transpose()

x_testi_samples = get_steps_btwn(encoder  = enc,
                                 decoder  = dec,
                                 report_a = report_a,
                                 report_b = report_b,
                                 df_sd    = sds,
                                 df_mn    = mns,
                                 steps    = 10,
                                 reinstate_neg = True,
                                 reinstate_nan = False
                                 )

x_testi_samples.to_csv(os.path.join(proj_root, 'x_testi_'+file_short+'_samples_far.csv'), mode='w', header=True, index=True)


