import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


hdf = pd.HDFStore('Train.h5', mode='r')
train = hdf.get('/df')
train = train.fillna(0)



def count_words(s):
   s = str(s)
   return len(s.split())

train['# of tweet content'] = train['Tweet content'].apply(count_words)
      
#plt.scatter(train['Favs'], train['rank'])
#plt.xlabel("Favs")
#plt.ylabel("rank")

train = train[train['Favs'] < 200]

#plt.scatter(train['RTs'], train['rank'])
#plt.xlabel("RTs")
#plt.ylabel("rank")

#(train['RTs'] > 20000).count()
#m = train['RTs']

train = train[train['RTs'] < 20000]

#plt.scatter(train['Followers'], train['rank'])
#plt.xlabel("Followers")
#plt.ylabel("rank")

train = train[train['Followers'] < 1000000]

#plt.scatter(train['Following'], train['rank'])
#plt.xlabel("Following")
#plt.ylabel("rank")

train = train[train['Following'] < 160000]

#plt.scatter(train['Listed'], train['rank'])
#plt.xlabel("Listed")
#plt.ylabel("rank")

train = train[train['Listed'] < 15000]

#plt.scatter(train['likes'], train['rank'])
#plt.xlabel("likes")
#plt.ylabel("rank")

train = train[train['likes'] < 80000]

#plt.scatter(train['tweets'], train['rank'])
#plt.xlabel("tweets")
#plt.ylabel("rank")

train = train[train['tweets'] < 30000]

#plt.scatter(train['reply'], train['rank'])
#plt.xlabel("reply")
#plt.ylabel("rank")

train = train[train['reply'] < 2500]




