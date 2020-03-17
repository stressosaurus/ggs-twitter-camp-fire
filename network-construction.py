#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import json
import myUtilities as mu

# tweets information
T = pd.DataFrame(np.load('data-subset/CAF-words-fire-related-words-tweets.npy',allow_pickle=True).item())
# user information
U = np.load('data-subset/CAF-words-fire-related-words-users.npy',allow_pickle=True).item()
# bot user information
B = np.load('data-subset/CAF-words-fire-related-words-bscore.npy',allow_pickle=True).item()
# networks
N = np.load('data-subset/CAF-words-fire-related-words-network.npy',allow_pickle=True).item()

# separate tweets by time, November 2018 and November 2019.
# November 2018
T_112018 = mu.get_subset_ymd(T,ymd=(2018,11,None))
T_112018_USN = list(T_112018['user_screen_name'].values)
for i in T_112018['usermentions'].values:
    if i != '*':
        T_112018_USN.extend(i.split(','))
T_112018_USN = np.unique(T_112018_USN)
T_112018_HTGS = []
for i in T_112018['hashtags'].values:
    if i != '*':
        T_112018_HTGS.extend(i.lower().split(','))
T_112018_HTGS = np.unique(T_112018_HTGS)

# November 2019
T_112019 = mu.get_subset_ymd(T,ymd=(2019,11,None))
T_112019_USN = list(T_112019['user_screen_name'].values)
for i in T_112019['usermentions'].values:
    if i != '*':
        T_112019_USN.extend(i.split(','))
T_112019_USN = np.unique(T_112019_USN)
T_112019_HTGS = []
for i in T_112019['hashtags'].values:
    if i != '*':
        T_112019_HTGS.extend(i.lower().split(','))
T_112019_HTGS = np.unique(T_112019_HTGS)

# create directories to save the networks
try:
    os.mkdir('data-networks')
except:
    pass

# categorizes the bot score into one of 3 groups
def bot_3categorizer(b):
    if b < 1.6666:
        return 1
    elif b >= 1.6666 and b <= 3.3332:
        return 2
    elif b > 3.3332:
        return 3
    else:
        return -1 # this means no info

# categorizes the bot score into on of 5 groups
def bot_5categorizer(b):
    if b < 1:
        return 1
    elif b >= 1 and b < 2:
        return 2
    elif b >= 2 and b < 3:
        return 3
    elif b >= 3 and b < 4:
        return 4
    elif b >= 4:
        return 5
    else:
        return -1 # this means no info
     
### November 2018        
        
# users cooccurence network
print('processing users network...')
G_USN_main = nx.Graph()
G_USN_main.add_edges_from(N['users-coocurrence'].keys())
G_USN = G_USN_main.subgraph(T_112018_USN)

degree = nx.degree_centrality(G_USN)
centrality = nx.eigenvector_centrality(G_USN,max_iter=1000,weight='frequency')
betweenness = nx.betweenness_centrality(G_USN,weight='frequency')
nx.set_node_attributes(G_USN,degree,'degree')
nx.set_node_attributes(G_USN,centrality,'centrality')
nx.set_node_attributes(G_USN,betweenness,'betweenness')
print('computing centralities done...')

node_bot_3cat = {}
node_bot_5cat = {}
for k, j in enumerate(G_USN.nodes()):
    usn = U['key'][j]
    try:
        usn_nf = U['information'][usn]['user_number_of_followers']
    except:
        usn_nf = -1
    try:
        bot_metric = (1/2)*5*B[j]['cap-english']+(1/2)*B[j]['display_scores-english']
    except:
        bot_metric = -1
    node_bot_3cat[j] = bot_3categorizer(bot_metric)
    node_bot_5cat[j] = bot_5categorizer(bot_metric)
nx.set_node_attributes(G_USN,node_bot_3cat,'bot_3cat')
nx.set_node_attributes(G_USN,node_bot_5cat,'bot_5cat')
print('node attributes populated...')

edge_freq = {}
for j in G_USN.edges():
    value = 1
    try:
        value = N['users-coocurrence'][j]['frequency']
    except:
        pass
    try:
        j_rev = list(reversed(j))
        value += N['users-coocurrence'][(j_rev[0],j_rev[1])]['frequency']
    except:
        pass
    edge_freq[j] = value
nx.set_edge_attributes(G_USN,edge_freq,'frequency')
print('edge attributes populated...')
nx.write_gpickle(G_USN,'data-networks/USN-nx-112018.gpickle')

# users - get connected components
print('getting components...')
G_USN_comps = list(nx.connected_components(G_USN))
G_USN_comps_sizes = [len(i) for i in G_USN_comps]
G_USN_comps_sorted = [G_USN_comps[i] for i in np.argsort(-1*np.array(G_USN_comps_sizes))]
USN_COMP_G = []
for I, i in enumerate(G_USN_comps_sorted):
    USN_COMP_G.append(G_USN.subgraph(i))
nx.write_gpickle(USN_COMP_G,'data-networks/USN-nx-112018-comps.gpickle')
print()

# hashtags cooccurence network
print('processing hashtags network...')
G_HTGS_main = nx.Graph()
G_HTGS_main.add_edges_from(N['hashtags-coocurrence'].keys())
G_HTGS = G_HTGS_main.subgraph(T_112018_HTGS)

degree = nx.degree_centrality(G_HTGS)
centrality = nx.eigenvector_centrality(G_HTGS,max_iter=1000,weight='frequency')
betweenness = nx.betweenness_centrality(G_HTGS,weight='frequency')
nx.set_node_attributes(G_HTGS,degree,'degree')
nx.set_node_attributes(G_HTGS,centrality,'centrality')
nx.set_node_attributes(G_HTGS,betweenness,'betweenness')
print('computing centralities done...')

edge_freq = {}
for j in G_HTGS.edges():
    value = 1
    try:
        value = N['hashtags-coocurrence'][j]['frequency']
    except:
        pass
    try:
        j_rev = list(reversed(j))
        value += N['hashtags-coocurrence'][(j_rev[0],j_rev[1])]['frequency']
    except:
        pass
    edge_freq[j] = value
nx.set_edge_attributes(G_HTGS,edge_freq,'frequency')
print('edge attributes populated...')
nx.write_gpickle(G_HTGS,'data-networks/HTGS-nx-112018.gpickle')
    
# hashtags - get connected components
G_HTGS_comps = list(nx.connected_components(G_HTGS))
G_HTGS_comps_sizes = [len(i) for i in G_HTGS_comps]
G_HTGS_comps_sorted = [G_HTGS_comps[i] for i in np.argsort(-1*np.array(G_HTGS_comps_sizes))]

HTGS_COMP_G = []
for I, i in enumerate(G_HTGS_comps_sorted):
    HTGS_COMP_G.append(G_HTGS.subgraph(i))
nx.write_gpickle(HTGS_COMP_G,'data-networks/HTGS-nx-112018-comps.gpickle')
print()


### November 2019

# users cooccurence network
print('processing users network...')
G_USN_main = nx.Graph()
G_USN_main.add_edges_from(N['users-coocurrence'].keys())
G_USN = G_USN_main.subgraph(T_112019_USN)

degree = nx.degree_centrality(G_USN)
centrality = nx.eigenvector_centrality(G_USN,max_iter=1000,weight='frequency')
betweenness = nx.betweenness_centrality(G_USN,weight='frequency')
nx.set_node_attributes(G_USN,degree,'degree')
nx.set_node_attributes(G_USN,centrality,'centrality')
nx.set_node_attributes(G_USN,betweenness,'betweenness')
print('computing centralities done...')

node_bot_3cat = {}
node_bot_5cat = {}
for k, j in enumerate(G_USN.nodes()):
    usn = U['key'][j]
    try:
        usn_nf = U['information'][usn]['user_number_of_followers']
    except:
        usn_nf = -1
    try:
        bot_metric = (1/2)*5*B[j]['cap-english']+(1/2)*B[j]['display_scores-english']
    except:
        bot_metric = -1
    node_bot_3cat[j] = bot_3categorizer(bot_metric)
    node_bot_5cat[j] = bot_5categorizer(bot_metric)
nx.set_node_attributes(G_USN,node_bot_3cat,'bot_3cat')
nx.set_node_attributes(G_USN,node_bot_5cat,'bot_5cat')
print('node attributes populated...')

edge_freq = {}
for j in G_USN.edges():
    value = 1
    try:
        value = N['users-coocurrence'][j]['frequency']
    except:
        pass
    try:
        j_rev = list(reversed(j))
        value += N['users-coocurrence'][(j_rev[0],j_rev[1])]['frequency']
    except:
        pass
    edge_freq[j] = value
nx.set_edge_attributes(G_USN,edge_freq,'frequency')
print('edge attributes populated...')
nx.write_gpickle(G_USN,'data-networks/USN-nx-112019.gpickle')

# users - get connected components
print('getting components...')
G_USN_comps = list(nx.connected_components(G_USN))
G_USN_comps_sizes = [len(i) for i in G_USN_comps]
G_USN_comps_sorted = [G_USN_comps[i] for i in np.argsort(-1*np.array(G_USN_comps_sizes))]
USN_COMP_G = []
for I, i in enumerate(G_USN_comps_sorted):
    USN_COMP_G.append(G_USN.subgraph(i))
nx.write_gpickle(USN_COMP_G,'data-networks/USN-nx-112019-comps.gpickle')
print()

# hashtags cooccurence network
print('processing hashtags network...')
G_HTGS_main = nx.Graph()
G_HTGS_main.add_edges_from(N['hashtags-coocurrence'].keys())
G_HTGS = G_HTGS_main.subgraph(T_112019_HTGS)

degree = nx.degree_centrality(G_HTGS)
centrality = nx.eigenvector_centrality(G_HTGS,max_iter=1000,weight='frequency')
betweenness = nx.betweenness_centrality(G_HTGS,weight='frequency')
nx.set_node_attributes(G_HTGS,degree,'degree')
nx.set_node_attributes(G_HTGS,centrality,'centrality')
nx.set_node_attributes(G_HTGS,betweenness,'betweenness')
print('computing centralities done...')

edge_freq = {}
for j in G_HTGS.edges():
    value = 1
    try:
        value = N['hashtags-coocurrence'][j]['frequency']
    except:
        pass
    try:
        j_rev = list(reversed(j))
        value += N['hashtags-coocurrence'][(j_rev[0],j_rev[1])]['frequency']
    except:
        pass
    edge_freq[j] = value
nx.set_edge_attributes(G_HTGS,edge_freq,'frequency')
print('edge attributes populated...')
nx.write_gpickle(G_HTGS,'data-networks/HTGS-nx-112019.gpickle')
    
# hashtags - get connected components
G_HTGS_comps = list(nx.connected_components(G_HTGS))
G_HTGS_comps_sizes = [len(i) for i in G_HTGS_comps]
G_HTGS_comps_sorted = [G_HTGS_comps[i] for i in np.argsort(-1*np.array(G_HTGS_comps_sizes))]

HTGS_COMP_G = []
for I, i in enumerate(G_HTGS_comps_sorted):
    HTGS_COMP_G.append(G_HTGS.subgraph(i))
nx.write_gpickle(HTGS_COMP_G,'data-networks/HTGS-nx-112019-comps.gpickle')
print()