#!/usr/bin/env python 

import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
import networkx as nx
import os
import myUtilities as mu

try:
    os.mkdir('data-subset')
except:
    pass

print('Loading datasets...')

# tweets information
T = pd.DataFrame(np.load('data/words-fire-related-words-tweets.npy',allow_pickle=True).item()).T

# user information
U = np.load('data/words-fire-related-words-users.npy',allow_pickle=True).item()

# networks
N = np.load('data/words-fire-related-words-network.npy',allow_pickle=True).item()

# list seed words and hashtags
print('Listing seed words...')
seed_wrds_CAF = []
with open('CAF-seed-word-list.txt') as f:
    for i in f:
        seed_wrds_CAF.append(i.replace('\n',''))
seed_wrds_CAF = set(seed_wrds_CAF)
seed_htgs_CAF_0 = []
with open('CAF-seed-hashtag-list.txt') as f:
    for i in f:
        seed_htgs_CAF_0.append(i.replace('\n',''))
seed_htgs_CAF_1 = []
for i in N['hashtags-coocurrence'].keys():
    if len(set(i) & set(seed_htgs_CAF_0)) != 0:
        seed_htgs_CAF_1.extend(i)
seed_htgs_CAF = set(seed_htgs_CAF_1)

# pick tweets ids with seed words and hashtags
print('Listing tweets ids...')
CAF_ids = []
for i in T['text'].keys():
    txt = T['text'][i].lower()
    if len(set(list(tknzr.tokenize(txt))) & seed_wrds_CAF) != 0:
        CAF_ids.append(i)
for i in T['hashtags'].keys():
    htgs = T['hashtags'][i].lower()
    if len(set(htgs.split(',')) & seed_htgs_CAF) != 0:
        CAF_ids.append(i)
        
# construct tweet-reply network to get related tweets
print('Listing relevant tweets ids...')
edges = list(N['tweet-reply'].keys())
G = nx.Graph()
G.add_edges_from(edges)

# using depth-first search algorithm to traverse through the tree network using the seed nodes
all_CAF_ids = []
for i in CAF_ids:
    all_CAF_ids.append(i)
    try:
        pred = nx.dfs_predecessors(G, source=i)
        all_CAF_ids.extend(list(pred.values()))
    except:
        pass
    try:
        succ = nx.dfs_successors(G, source=i)
        for j in succ.keys():
            all_CAF_ids.append(j)
            all_CAF_ids.extend(succ[j])
    except:
        pass
all_CAF_ids = np.unique(all_CAF_ids)
del G

# subset Tweets
print('Subsetting tweets...')
TIDS_main = T['parent_tweet_id'].keys()
exists = list(set(TIDS_main) & set(all_CAF_ids))
T_sub_all = T.loc[exists,:]
T_subs = [mu.get_subset_ymd(T_sub_all,ymd=(2018,11,None)),mu.get_subset_ymd(T_sub_all,ymd=(2019,11,None))]
T_sub = pd.concat(T_subs)

np.save('data-subset/CAF-words-fire-related-words-tweets.npy',T_sub.to_dict())

# subset users
print('Subsetting users...')
users_codes = list(np.unique(T_sub['user_screen_name'].values))
users_mentions = T_sub['usermentions'].values
for i in users_mentions:
    if i != '*':
        users_codes.extend(i.split(','))
U_sub = {}
U_sub['key'] = {}
U_sub['information'] = {}
for i in users_codes:
    usn = U['key'][i]
    U_sub['key'][i] = usn
    U_sub['key'][usn] = i
    U_sub['information'][usn] = U['information'][usn]
np.save('data-subset/CAF-words-fire-related-words-users.npy',U_sub)
print(len(U_sub['information'].keys()))

# subset networks
print('Subsetting networks...')
N_sub = {}
N_sub['tweet-reply'] = {}
N_sub['hashtags-coocurrence'] = {}
N_sub['users-coocurrence'] = {}
TIDS = list(T_sub['parent_tweet_id'].keys())
for i in TIDS:
    PTID = T_sub['parent_tweet_id'][i]
    QTID = T_sub['quoted_tweet_id'][i]
    if PTID != '*':
        ptid_edge = (int(PTID),int(i))
        N_sub['tweet-reply'][ptid_edge] = N['tweet-reply'][ptid_edge]
    if PTID == '*' and QTID != '*':
        qtid_edge = (int(QTID),int(i))
        N_sub['tweet-reply'][qtid_edge] = N['tweet-reply'][qtid_edge]
        
    htgs = T_sub['hashtags'][i].lower().split(',')
    if htgs != ['*']:
        for j_i, j_j in enumerate(htgs):
            for k in htgs[j_i:]:
                if j_j != k:
                    try:
                        N_sub['hashtags-coocurrence'][(j_j,k)]['frequency'] += 1
                    except:
                        N_sub['hashtags-coocurrence'][(j_j,k)] = {'frequency':1}
    
    ums = T_sub['usermentions'][i].split(',')
    if ums != ['*']:
        for j_i, j_j in enumerate(ums):
            for k in ums[j_i:]:
                if j_j != k:
                    try:
                        N_sub['users-coocurrence'][(j_j,k)]['frequency'] += 1
                    except:
                        N_sub['users-coocurrence'][(j_j,k)] = {'frequency':1}
np.save('data-subset/CAF-words-fire-related-words-network.npy',N_sub)
print(len(N_sub['tweet-reply'].keys()))
print(len(N_sub['hashtags-coocurrence'].keys()))
print(len(N_sub['users-coocurrence'].keys()))
                        
print('Done!')