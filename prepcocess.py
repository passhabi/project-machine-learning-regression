import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from learnt import regression as lr
from scipy import stats
from sklearn.metrics import mean_squared_error as mse_error
from sklearn.model_selection import train_test_split


def preprocessing():
    # importing the data set, at same time we do same for StudentTest:
    dtype_dict = {'Tweet Id': int, 'User Name': str, 'Favs': int, 'RTs': int, 'Followers': int,
                  'Following': int, 'Listed': int, 'likes': int, 'tweets': int, 'reply': int,
                  'URLs': str, 'Tweet content': str}

    #   import:
    df = pd.read_hdf('Train.h5', key='/df', mode='r')
    StudentTest = pd.read_excel('StudentTest.xlsx', converters=dtype_dict)

    # change data and handle missing values:
    #   drop useless rows:
    df = df.dropna(subset=['rank'], how='any')
    df = df.drop(252)  # its a useless row

    #   how many nans we have in data frame now:
    # df_info = pd.DataFrame()
    # df_info['percentage'] = (df.isnull().sum() * 100.0) / len(df)

    #   let's fill nans values with 0 as the problem description demanded.
    df['Listed'] = df['Listed'].fillna(1)  # little cheat (maybe), filling Listed nans with most frequent occur value
    df = df.fillna(0)  # fill other nans with zero
    #       do the same for StudentTest:
    StudentTest['Listed'] = StudentTest['Listed'].fillna(1)
    StudentTest = StudentTest.fillna(0)

    #   change the float data type to int (Only for the considered columns):
    int_col_list = ['Favs', 'RTs', 'Followers', 'Following', 'Listed', 'likes', 'tweets', 'reply', 'rank']
    df[int_col_list] = df[int_col_list].astype(int)

    # lets add a new input, called 'word count':
    df['word count'] = df['Tweet content'].apply(lambda sentence: len(str(sentence).split()))
    StudentTest['word count'] = StudentTest['Tweet content'].apply(lambda sentence: len(str(sentence).split()))

    # lets drop the useless columns too:
    df = df.drop(columns=['Tweet Id','User Name', 'Favs', 'RTs', 'URLs','Tweet content'])
    StudentTest = StudentTest.drop(columns=['Tweet Id', 'User Name', 'Favs', 'RTs', 'URLs','Tweet content'])

    df = df.reset_index(drop=True)   # update data frame indices

    # lets look for outliers and delete them:
    n = len(df)  # number of data point before deleting outliers
    print(len(df), 'data point are in data frame')

    # rank :
    # ----------------------------------------------
    threshold = 7
    z_scores = np.abs(stats.zscore(df['rank']))
    # col = np.where(z_scores > threshold, 'r', 'b')
    # plt.scatter(df['Followers'], df['rank'], c=col, marker='.')
    # plt.show()

    #   deleting outliers form data frame:
    indices = np.where(z_scores > threshold)[0]  # get the outlier corresponding indices
    df = df.drop(index=indices)  # drop the outliers

    # Followers:
    # ----------------------------------------------
    x = 'Followers'
    # plt.scatter(df[x], df['rank'], marker='.')
    # plt.xlim([100000, 500000])
    # plt.show()
    print(len(df[df[x] > 300000]), 'data points deleted from data frame due to Followers outlier removing.')
    df = df[df[x] < 300000]

    # indices = df[df[x] > 300000].index
    # max is : 300000
    # df.at[indices, x] = 300000  # impute outliers to the max 300000 of the none outliers.

    # Following:
    # ----------------------------------------------
    x = 'Following'
    # plt.scatter(df[x], df['rank'], marker='.')
    # plt.show()
    # print(len(df[df[x] > 65000]), 'data points deleted from data frame due to Following outlier removing.')
    df = df[df[x] < 75000]

    # indices = df[df[x] > 65000].index
    # max is around: 65000
    # df.at[indices, x] = 65000  # impute outliers to the max 65000 of the none outliers.

    # removing Listed outlier:
    # ----------------------------------------------
    x = 'Listed'
    # plt.scatter(df[x], df['rank'], marker='.')
    # plt.show()
    print(len(df[df[x] > 75000]), 'data points deleted from data frame due to Listed outlier removing.')
    df = df[df[x] < 75000]

    # indices = df[df[x] > 12500].index
    # max is around: 12500
    # df.at[indices, x] = 12500  # impute outliers to the max 12500 of the none outliers.

    # removing likes outlier:
    # ----------------------------------------------
    x = 'likes'
    # plt.scatter(df[x], df['rank'], marker='.')
    # plt.show()
    print(len(df[df[x] > 80000]), 'data points deleted from data frame due to likes outlier removing.')
    # df = df[df[x] < 80000]

    # there is a diff between outliers, trying in to keep the range (distance):
    high_indices = df[df[x] > 70000].index

    indices = df[df[x] > 60000].index
    # max is around: 35000
    df.at[indices, x] = 40000  # impute lower outliers to the max 40000 of the none outliers.
    df.at[high_indices, x] = 45000  # impute higher outliers to the max 45000 of the none outliers.

    # removing tweets outlier:
    # ----------------------------------------------
    # no outlier

    # removing reply outlier:
    # ----------------------------------------------
    # x = 'reply'
    # plt.scatter(df[x], df['rank'], marker='.')
    # plt.show()
    # not sure to delete any

    # removing reply outlier:
    # ----------------------------------------------
    # no outlier

    print('total deleted data point:', n - len(df))


    # Adding our own features:
    df['Followers_sub_Following'] = df['Followers'] - df['Following']
    df['tweets_reply'] = df['tweets'] * df['reply']
    df['log_likes'] = df['likes'].apply(lambda x: np.log(x))

    StudentTest['Followers_sub_Following'] = StudentTest['Followers'] - StudentTest['Following']
    StudentTest['tweets_reply'] = StudentTest['tweets'] * StudentTest['reply']
    StudentTest['log_likes'] = StudentTest['likes'].apply(lambda x: np.log(x))

    return df, StudentTest
