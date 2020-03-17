import numpy as np
import pandas as pd
import re
import string
import os
import torch

# BERT - mark given text with [CLS] and [SEP]
def mark_tweet(x):
    return '[CLS] '+x+' [SEP]'

# BERT - tokenize and get indices of given text
def tokenize(x,tokenizer):
    x_tokens = tokenizer.tokenize(x)
    x_index = tokenizer.convert_tokens_to_ids(x_tokens)
    x_segments = [1]*len(x_index)
    return x_tokens, x_index, x_segments

# BERT - construct tweet embedding by averaging the columns of each layer (from BERT) and stacking them
def make_tweet_embedding_stk(x):

    # take mean of each layer
    means = []
    for layer_i in x:
        means.append(torch.mean(layer_i[0],0))

    # stack means
    stacked_means = []
    for m in means:
        stacked_means.extend(m.detach().numpy())

    return np.array(stacked_means)

# BERT - construct tweet embedding by taking the first row ([CLS] token) of the last layer (from BERT)
def make_tweet_embedding_cls(x):

    # take last layer
    last_layer = x[len(x)-1]

    # take first row
    first_row = last_layer[0][0,:].detach().numpy()

    return np.array(first_row)

# get tweet data set
def get_tweets_data(TYPE,LABEL):
    file_t = TYPE+'-'+LABEL+'-tweets.npy'
    file_u = TYPE+'-'+LABEL+'-users.npy'
    data_path = '/raw-data/twitter-tweets-scrape/'+TYPE+'-processed/'
    root_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    full_path_t = root_path+data_path+file_t
    T = pd.DataFrame(np.load(full_path_t,allow_pickle=True).item()).T

    full_path_u = root_path+data_path+file_u
    U = np.load(full_path_u,allow_pickle=True).item()

    return T, U

# get subset of tweets by year, month, and/or day
def get_subset_ymd(T,ymd=[None,None,None]):
    if ymd[0] != None:
        year_criteria = T['year'] == ymd[0]
        T = T.loc[year_criteria]
    if ymd[1] != None:
        month_criteria = T['month'] == ymd[1]
        T = T.loc[month_criteria]
    if ymd[2] != None:
        day_criteria = T['day'] == ymd[2]
        T = T.loc[day_criteria]

    return T

# get users and user_mentions from text
def get_users(x):
    x_out = x.replace('@ ','@')
    x_out_um = re.findall(r'(@[\w_-]+)',x_out)
    return x_out_um

# get hashtags from text
def get_hashtags(x):
    x_out = x.replace('# ','#')
    x_out_um = re.findall(r'(#[\w_-]+)',x_out)
    return x_out_um

# remove users and user_mentions from text
def remove_users(x):
    x_out = re.sub(r'\[[a-zA-Z]+-USN[0-9]+\]','',x)
    x_out = re.sub(r'\s+',' ',x_out)
    return x_out

# remove hashtags from text
def remove_hashtags(x):
    x_out = re.sub(r'(#[\w_-]+)','',x)
    x_out = re.sub(r'\s+',' ',x_out)
    return x_out

# text cleaner (individual)
def tweet_text_cleaner(x):
    # remove other characters
    x_out = x.replace('...','')
    x_out = x_out.replace('…','')
    x_out = x_out.replace('..','')
    x_out = x_out.replace('.. ..','')
    # remove newline characters
    x_out = re.sub(r'\n+',' ',x_out)
    # remove hyperlinks
    x_out = re.sub(r'http:\/\/\s','http://',x_out)
    x_out = re.sub(r'https:\/\/\s','https://',x_out)
    x_out = re.sub(r'https?:\/\/.*\/\w*','',x_out)
    # remove twitter picture hyperlinks
    x_out = re.sub(r'pic.twitter.com\/\w*','',x_out)
    # extract user mentions
    x_out = x_out.replace('@ ','@')
    x_out_um = re.findall(r'(@[\w_-]+)',x_out)
    # match hashtags
    x_out = x_out.replace('# ','#')
    x_out_ht = re.findall(r'(#[\w_-]+)',x_out)
    # replace multiple spaces into one space
    x_out = re.sub(' +',' ',x_out)

    return x_out, x_out_ht, x_out_um

# time splitter (individual)
def time_split(x,by='month'):
    x_split = x.split(' ')
    date = x_split[0]
    time = x_split[1]
    date_split = date.split('-')
    time_split = time.split(':')
    time_vect = date_split + time_split
    out_time = []
    if by == 'year':
        out_time = [int(i) for i in time_vect[0:1]]
    elif by == 'month':
        out_time = [int(i) for i in time_vect[0:2]]
    elif by == 'day':
        out_time = [int(i) for i in time_vect[0:3]]
    elif by == 'hour':
        out_time = [int(i) for i in time_vect[0:4]]
    elif by == 'minute':
        out_time = [int(i) for i in time_vect[0:5]]
    elif by == 'second':
        out_time = [int(i) for i in time_vect[0:6]]
    return out_time
