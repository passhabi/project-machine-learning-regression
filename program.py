import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from learnt import regression as lr
from scipy import stats

'pre processing data:'
# import data as a data frame called train:
hdf = pd.HDFStore('Train.h5', mode='r')
df = hdf.get('/df')

# 2. Counting number of words in a sentence:
df['# of Tweet content'] = df['Tweet content'].apply(lambda sentence:
                                                     len(str(sentence).split()))

# how many nans are in my data frame:
df_info = pd.DataFrame()
df_info['Number of Nans'] = df.isnull().sum()
df_info['empty percentage'] = (df_info['Number of Nans'] * 100.0) / len(df)

# 1. fill all nans in data frame with zero:
df = df.fillna(0)

# 3. looking for outlier data points (high leverage points):
n = len(df)  # number of data point before deleting outliers
print(n, 'data point are in data frame')
# removing rank outlier:
# ----------------------------------------------
threshold = 7
z_scores = np.abs(stats.zscore(df['rank']))
# col = np.where(z_scores > threshold, 'r', 'b')
# plt.scatter(df['Favs'], df['rank'], c=col, marker='.')
# plt.show()

# deleting outliers form data frame:
indices = np.where(z_scores > threshold)[0]  # get the outlier corresponding indices
df = df.drop(index=indices)  # drop the outliers

# removing Favs outlier:
# ----------------------------------------------
x = 'Favs'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.show()
print(len(df[df[x] > 70]), 'data points will be deleted from data frame due to Favs outlier removing.')
df = df[df[x] < 70]

# removing RTs outlier:
# ----------------------------------------------
x = 'RTs'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.show()
print(len(df[df[x] > 20000]), 'data points deleted from data frame due to RTs outlier removing.')
df = df[df[x] < 20000]

# removing Followers outlier:
# ----------------------------------------------
x = 'Followers'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.show()
print(len(df[df[x] > 300000]), 'data points deleted from data frame due to Followers outlier removing.')
df = df[df[x] < 300000]

# removing Following outlier:
# ----------------------------------------------
x = 'Following'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.show()
print(len(df[df[x] > 160000]), 'data points deleted from data frame due to Following outlier removing.')
df = df[df[x] < 160000]

# removing Listed outlier:
# ----------------------------------------------
x = 'Listed'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.show()
print(len(df[df[x] > 15000]), 'data points deleted from data frame due to Listed outlier removing.')
df = df[df[x] < 15000]

# removing likes outlier:
# ----------------------------------------------
x = 'likes'
plt.scatter(df[x], df['rank'], marker='.')
plt.show()
print(len(df[df[x] > 80000]), 'data points deleted from data frame due to likes outlier removing.')
df = df[df[x] < 80000]

# removing tweets outlier:
# ----------------------------------------------
outlier_selection = lr.OutlierDetectHelper()
outlier_selection.outlier_selection_by_linear_model(df, 'tweets', 'rank')
# outlier_selection.plot_excluded_outliers(k=1)
index_to_delete = outlier_selection.get_k_outlier(1)[0]
df = df.drop(index=index_to_delete)
print('1 data points deleted from data frame due to tweets outlier removing.')

# removing reply outlier:
# ----------------------------------------------
x = 'reply'
plt.scatter(df[x], df['rank'], marker='.')
plt.show()
# not sure to delete any

# removing reply outlier:
# ----------------------------------------------
# no outlier

print('total deleted data point:', n - len(df))

# what features we want to include in our model:
features_list = ['RTs', 'Followers', 'Following',
                 'Listed', 'likes', 'tweets', 'reply', '# of Tweet content']
# split data frame to test and training set:
x, y = lr.get_numpy_data(df, features_list, 'rank')
from sklearn.model_selection import train_test_split

features, features_test, output, output_test = train_test_split(x, y, test_size=0.25, random_state=42)
