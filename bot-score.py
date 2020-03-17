#!/usr/bin/env python

#PREAMBLE
import botometer
import numpy as np
from time import sleep
import ast

#LOAD DATA
print('Loading dataset...')
U = np.load('data-subset/CAF-words-fire-related-words-users.npy').item()

#GIVES VECTOR OF USER ID'S
unique = list(U['information'].keys())

#INPUTTING API VALIDATIONS
rapidapi_key = "017ad380c9mshc0ccb7d4ab5adc2p1527dajsncc9c0a43699d" # now it's called rapidapi key
twitter_app_auth = {
    'consumer_key': '37DwzL2QSE8mJFIde9v7KnTBq',
    'consumer_secret': '7fcql5oqG1y7dwGdSemlfAfPzSNNkiSMrpSHGqVEQZUfpZQbqR',
    'access_token': '1093278124224131072-rf4kg7PvpCb3joaJYg23yUwXXlvr5i',
    'access_token_secret': '9iWRvJlSaOVWOp72D3GVr503aXFb3V8HCWPZxxi9eVJrl',
  }
bom = botometer.Botometer(wait_on_ratelimit=True,
                          rapidapi_key=rapidapi_key,
                          **twitter_app_auth)

#DEFINE FUNCTION, RETURNS SCORE DATA
def bot_scores(users):
    r = bom.check_accounts_in(users)
    return r

#CREATING TEXT FILE OF THE SCORES
print('requesting scores...')
f= open("data-subset/bscores-raw.txt","w+")

# request scores using API
bscores_results = bot_scores(unique)

for a, s in bscores_results:
    f.write(str({a:s}) + "\n")
f.close()

bscores = {}
with open('data-subset/bscores-raw.txt') as read_file:
    for i in read_file:
        line = ast.literal_eval(i)
        try:
            uid_local = list(line.keys())[0]
            patch_local = U['key'][uid_local]
            bscores[patch_local] = {}
            for i in line[uid_local].keys():
                if i != 'user':
                    for j in line[uid_local][i].keys():
                        if j != 'user':
                            bscores[patch_local][i+'-'+j] = line[uid_local][i][j]
        except:
            pass
np.save('data-subset/CAF-words-fire-related-words-bscore.npy',bscores)

print('Done!')
