import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from learnt import regression as lr
from scipy import stats
from sklearn.metrics import mean_squared_error as mse_error
from sklearn.model_selection import train_test_split

'pre processing'
# import data as a data frame called train:
hdf = pd.HDFStore('Train.h5', mode='r')
df = hdf.get(hdf.keys()[0])

# df = pd.read_hdf('Train.h5', '/df')

# 2. Counting number of words in a sentence:
df['# of Tweet content'] = df['Tweet content'].apply(lambda sentence:
                                                     len(str(sentence).split()))

# how many nans are in my data frame:
# df_info = pd.DataFrame()
# df_info['Number of Nans'] = df.isnull().sum()
# df_info['empty percentage'] = (df_info['Number of Nans'] * 100.0) / len(df)

# 1. fill all nans in data frame with zero:
df = df.fillna(0)

# 3. looking for outlier data points (high leverage points):
n = len(df)  # number of data point before deleting outliers
print(n, 'data point are in data frame')

# removing rank outlier:
# ----------------------------------------------
threshold = 7
z_scores = np.abs(stats.zscore(df['rank']))
col = np.where(z_scores > threshold, 'r', 'b')
plt.scatter(df['Favs'], df['rank'], c=col, marker='.')
plt.show()

# deleting outliers form data frame:
indices = np.where(z_scores > threshold)[0]  # get the outlier corresponding indices
# df = df.drop(index=indices)  # drop the outliers
df.at[indices, 'rank'] = 1e2  # impute outliers to the max (1e2) of the none outliers.

# removing Favs outlier:
# ----------------------------------------------
x = 'Favs'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.xlim([0, 100])
# plt.show()
print(len(df[df[x] > 70]), 'data points will be deleted from data frame due to Favs outlier removing.')
# df = df[df[x] < 70]

indices = df[df[x] > 70].index
# max is : 70
df.at[indices, x] = 70  # impute outliers to the max 70 of the none outliers.

# removing RTs outlier:
# ----------------------------------------------
x = 'RTs'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.xlim([100, 5000])
# plt.show()
print(len(df[df[x] > 20000]), 'data points deleted from data frame due to RTs outlier removing.')
# df = df[df[x] < 20000]

indices = df[df[x] > 20000].index
# max is : 3000
df.at[indices, x] = 3000  # impute outliers to the max 3000 of the none outliers.

# removing Followers outlier:
# ----------------------------------------------
x = 'Followers'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.xlim([100000, 500000])
# plt.show()
print(len(df[df[x] > 300000]), 'data points deleted from data frame due to Followers outlier removing.')
# df = df[df[x] < 300000]

indices = df[df[x] > 300000].index
# max is : 300000
df.at[indices, x] = 300000  # impute outliers to the max 300000 of the none outliers.

# removing Following outlier:
# ----------------------------------------------
x = 'Following'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.show()
print(len(df[df[x] > 65000]), 'data points deleted from data frame due to Following outlier removing.')
# df = df[df[x] < 75000]

indices = df[df[x] > 65000].index
# max is around: 65000
df.at[indices, x] = 65000  # impute outliers to the max 65000 of the none outliers.

# removing Listed outlier:
# ----------------------------------------------
x = 'Listed'
# plt.scatter(df[x], df['rank'], marker='.')
# plt.show()
print(len(df[df[x] > 12500]), 'data points deleted from data frame due to Listed outlier removing.')
# df = df[df[x] < 12500]

indices = df[df[x] > 12500].index
# max is around: 12500
df.at[indices, x] = 12500  # impute outliers to the max 12500 of the none outliers.

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

# what features we want to include in our model:
features_list = ['Favs', 'RTs', 'Followers', 'Following',
                 'Listed', 'likes', 'tweets', 'reply', '# of Tweet content']


# B) 1.Which input, as feature gives the best simple model regression?
# split data frame to test and training set:
x, y = lr.get_numpy_data(df, features_list, 'rank')

# x = np.sort(x, 0)  # sort feature for visualisation along the columns
output = np.reshape(y, [-1, 1])
D = len(features_list)
mse = np.zeros(D - 1)
for i in range(1, D):
    feature = x[:, [0, i]]  # intercept + ith feature
    intercept, slope = lr.simple_linear_regression(feature, output)
    pred = lr.predict_outcome(feature, [intercept, slope])
    mse[i - 1] = lr.get_residual_sum_squares(output, pred)
    # plt.scatter(feature[:, 1], output)
    # plt.plot(feature[:, 1], pred.T, c='r')
    # plt.show()
    # plt.pause(2)
min_index = np.argmin(mse)  # index i correspond to feature with i
print('Best simple model made by [', features_list[min_index], '] feature')
print('its rss is:', min(mse))

'C) multiple regression'

# 1. Drop Highly Correlated Features:

# Create correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

df = df.drop(df.columns[to_drop], axis=1)

# split data to train, valid, test:
features, features_test, output, output_test = train_test_split(x, y, test_size=0.20, random_state=42)
features, features_valid, output, output_valid = train_test_split(features, output, test_size=0.20)

# 3. data normalization:
features, norm = lr.normalize_features(features)
features_valid = features_valid / norm
features_test = features_test / norm

initial_weights = np.zeros(len(features_list) + 1)
step_size = 4e-12
tolerance = 1e9
weights = lr.regression_gradient_descent(features, output, initial_weights, step_size, tolerance)

mse = mse_error(output, lr.predict_outcome(features, weights))
# train error 8166200974195.087
mse_test = mse_error(output_test, lr.predict_outcome(features_test, weights))
# test error 7305295832687.131


'E) lasso regression'
# cost_dict = {}
# for l1_penalty in np.logspace(0, 10, 5):
#     lasso_weights = lr.lasso_cyclical_coordinate_descent(features, output, initial_weights.tolist(),
#                                                           l1_penalty, 1.0)
#     pred = lr.predict_outcome(features_valid, lasso_weights)
#     cost_dict[l1_penalty] = mse_error(output_valid, pred)
#
# plt.plot(cost_dict.values(), '.-')
# plt.show()
#
# print(min(cost_dict.items(), key=lambda x: x[1]))
#
# # optimal lambda: 3562248
# l1_penalty = 1e7
# lasso_weights = lr.lasso_cyclical_coordinate_descent(features, output, initial_weights.tolist(), l1_penalty, 1.0)
# pred = lr.predict_outcome(features_valid, lasso_weights)
#
# # show what coefficient are zero:
# print('show what coefficients with optimal lambda value:')
# features_list_intercept = ['intercept'] + features_list
# for i in range(len(features_list_intercept)):
#     print(features_list_intercept[i], ': ', lasso_weights[i])
#
# print('mse error on train data', mse_error(output, lr.predict_outcome(features, lasso_weights)))
# print('mse error on test data', mse_error(output_test, lr.predict_outcome(features_test, lasso_weights)))


'D) ridge Regression'

features_list = ['Favs', 'RTs', 'Followers', 'Listed', 'likes', 'tweets', 'reply']

x, y = lr.get_numpy_data(df, features_list, 'rank')

# split data to train, valid, test:
features, features_test, output, output_test = train_test_split(x, y, test_size=0.20, random_state=43)
features, features_valid, output, output_valid = train_test_split(features, output, test_size=0.25, random_state=40)

# features = np.sort(features, 0)  # sort feature for visualisation along the columns

# normalise the data:
features, norm = lr.normalize_features(features)
features_valid = features_valid / norm
features_test = features_test / norm

initial_weights = np.zeros(len(features_list) + 1)
step_size = 4e-7
max_iteration = 100

from sklearn.linear_model import Ridge

cost_dict = {}
for l2_penalty in np.logspace(0, 2, 30):
    # weights = lr.ridge_regression_gradient_descent(features, output, initial_weights, step_size, l2_penalty,
    #                                                max_iteration)
    # pred = lr.predict_outcome(features_valid, weights)
    # fixme: the error with own ridge regression code
    model = Ridge(l2_penalty, fit_intercept=False)
    model.fit(features, output)
    pred = model.predict(features_valid)

    # plt.scatter(features[:, 5], output, marker='.')
    # plt.plot(features[:, 5], model.predict(features), c='r')
    # plt.xlabel("Tweets")
    # plt.ylabel('rank')
    # plt.show()
    # plt.pause(2.5)

    cost_dict[l2_penalty] = mse_error(output_valid, pred)

plt.plot(cost_dict.values(), '.-')
plt.xlabel('Mean square error')
plt.ylabel('l2_penalty')
plt.show()

print(min(cost_dict.items(), key=lambda x: x[1]))

# lambda has chosen as fallowing:
l2_penalty = 1.0
model = Ridge(l2_penalty, fit_intercept=False)
model.fit(features, output)
ridge_weights = model.coef_
pred = model.predict(features_valid)

print('mse error on train data', mse_error(output, lr.predict_outcome(features, ridge_weights)))
print('mse error on test data', mse_error(output_test, lr.predict_outcome(features_test, ridge_weights)))

''''''

# read the real test data:
# student_test_df = pd.read_excel('StudentTest.xlsx')
# student_test_df = student_test_df.fillna(0)
#
# features_list = ['Favs', 'RTs', 'Followers', 'Listed', 'likes', 'tweets', 'reply']
#
# x_student_test, _ = lr.get_numpy_data(student_test_df, features_list, 'Tweet Id')  # dont care about output.
#
# x_student_test = x_student_test/ norm

# predict Real Test with best model we so far found:
# x_student_test_pred = model.predict(x_student_test)
# x_student_test_pred = np.reshape(x_student_test_pred, [-1, 1])
#
# rank_df = pd.DataFrame(x_student_test_pred, columns=['rank'])
# student_test_df = student_test_df.join(rank_df)
#
# student_test_df.to_excel(r'\ridge model.xlsx', index = None, header=True)
'Forward step wise'

train_df = df.iloc[:1000, :]
test_df = df.iloc[1000:, :]

print(lr.forward_selected(train_df, test_df, features_list, 'rank'))

print(lr.backward_selected(train_df, test_df, features_list, 'rank'))

'1NN'
# nearest neighbor
oneNN = lr.knn.k_nearest_neighbors(1, features, features_test[0])
print('Prediction by 1NN for the 1st person in the data set is:', output[oneNN][0][0])
print('Real by 1NN for the 1st person  is:', output_test[0])

max_k = 200
mse_dict = {}
for k in range(1, max_k):
    valid_predict = lr.knn_predict_output(k, features, output, features_valid)
    mse_dict[k - 1] = mse_error(output_valid, valid_predict)  # mse start at 0

# lets see how k is:
plt.plot(range(1, max_k), mse_dict.values(), '.-')
plt.xlabel('K')
plt.ylabel('MSE')
plt.show()

print(min(mse_dict.items(), key=lambda x: x[1]))

# best value for k is 40:
test_predict = lr.knn_predict_output(40, features, output, features_test)
print('mse error on test data 40NN', mse_error(output_test, test_predict))


'''
the following commented codes are for saving the rank data in an execle file called "knn model":
'''
# hdf = pd.HDFStore('Train.h5', mode='r')
# df = hdf.get(hdf.keys()[0])
# df = df.fillna(0)
#
# df['# of Tweet content'] = df['Tweet content'].apply(lambda sentence: len(str(sentence).split()))
#
# features_list = ['Favs', 'RTs', 'Followers', 'Following',
#                  'Listed', 'likes', 'tweets', 'reply', '# of Tweet content']
#
# features, output = lr.get_numpy_data(df, features_list, 'rank')
#
# # read the real test data and save:
# student_test_df = pd.read_excel('StudentTest.xlsx')
# student_test_df = student_test_df.fillna(0)
#
# student_test_df['# of Tweet content'] = student_test_df['Tweet content'].apply(lambda sentence:
#                                                                                len(str(sentence).split()))
#
# x_student_test, _ = lr.get_numpy_data(student_test_df, features_list, 'Tweet Id')  # dont care about output.
#
# features, norm = lr.normalize_features(features)
# x_student_test = x_student_test / norm
#
#
# # predict Real Test with best model we so far found:
# x_student_test_pred = lr.knn_predict_output(40, features, output, x_student_test)
#
# rank_df = pd.DataFrame(x_student_test_pred, columns=['rank'])
# student_test_df = student_test_df.join(rank_df)
#
# student_test_df.to_excel(r'\knn model.xlsx', index=None, header=True)


x, y = lr.get_numpy_data(df, features_list, 'rank')

# split data to train, valid, test:
features, features_test, output, output_test = train_test_split(x, y, test_size=0.20, random_state=43)
features, features_valid, output, output_valid = train_test_split(features, output, test_size=0.25, random_state=40)

# features = np.sort(features, 0)  # sort feature for visualisation along the columns

# normalise the data:
features, norm = lr.normalize_features(features)
features_valid = features_valid / norm
features_test = features_test / norm


from sklearn.kernel_ridge import KernelRidge

model = KernelRidge(alpha=1)
model.fit(features, output)

predict = model.predict(features_valid)

print(mse_error(output_valid, predict))

test_predict = model.predict(features_test)

print(mse_error(output_test, test_predict))