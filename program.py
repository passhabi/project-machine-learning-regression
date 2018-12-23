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

feature_str = 'Following'
z_score = np.abs(stats.zscore(df[feature_str]))
col = np.where(z_score > 7, 'r', 'b')
plt.scatter(df[feature_str], df['rank'], c=col, marker='.')
plt.show()

outlier_selection = lr.OutlierDetectHelper()
outlier_selection.outlier_selection_by_linear_model(df, feature_str, 'rank')
outlier_selection.plot_excluded_outliers(k=100)

# what features we want to include in our model:
features_list = ['RTs', 'Followers', 'Following',
                 'Listed', 'likes', 'tweets', 'reply', '# of Tweet content']
# split data frame to test and training set:
x, y = lr.get_numpy_data(df, features_list, 'rank')
from sklearn.model_selection import train_test_split

features, features_test, output, output_test = train_test_split(x, y, test_size=0.25, random_state=42)
