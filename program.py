import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from learnt import regression as lr

'pre processing data:'
# import data as a data frame called train:
hdf = pd.HDFStore('Train.h5', mode='r')
df = hdf.get('/df')

# 1. fill all nans in data frame with zero:
df = df.fillna(0)


def count_words(sentence):
    """Counting number of words in a sentence
    """
    sentence = str(sentence)  # make sure we get a string content
    return len(sentence.split())


df['# of Tweet content'] = df['Tweet content'].apply(count_words)

# what features we want to include in our model:
features_list = ['Favs', 'RTs', 'Followers', 'Following',
                 'Listed', 'likes', 'tweets', 'reply', '# of Tweet content']
# split data frame to test and training set:
x, y = lr.get_numpy_data(df, features_list, 'rank')
from sklearn.model_selection import train_test_split

features, features_test, output, output_test = train_test_split(x, y, test_size=0.25, random_state=42)

# 2. looking for outlire data points (high leverage points):
plt.boxplot(features[:, 1])
plt.scatter(features[:, 1], output.reshape(-1, 1))
plt.show()

#
n = len(df[df['RTs'] > 20000])

df = df[df['Favs'] < 200]

plt.scatter(df['RTs'], df['rank'], marker='.')
plt.xlabel("RTs")
plt.ylabel("rank")
plt.show()

# (train['RTs'] > 20000).count()
# m = train['RTs']

df = df[df['RTs'] < 20000]

# plt.scatter(train['Followers'], train['rank'])
# plt.xlabel("Followers")
# plt.ylabel("rank")

df = df[df['Followers'] < 1000000]

# plt.scatter(train['Following'], train['rank'])
# plt.xlabel("Following")
# plt.ylabel("rank")

df = df[df['Following'] < 160000]

# plt.scatter(train['Listed'], train['rank'])
# plt.xlabel("Listed")
# plt.ylabel("rank")

df = df[df['Listed'] < 15000]

# plt.scatter(train['likes'], train['rank'])
# plt.xlabel("likes")
# plt.ylabel("rank")

df = df[df['likes'] < 80000]

# plt.scatter(train['tweets'], train['rank'])
# plt.xlabel("tweets")
# plt.ylabel("rank")

df = df[df['tweets'] < 30000]

# plt.scatter(train['reply'], train['rank'])
# plt.xlabel("reply")
# plt.ylabel("rank")

df = df[df['reply'] < 2500]
