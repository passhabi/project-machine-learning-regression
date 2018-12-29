from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse_error
import numpy as np


def forward_selected(df, df_test, features_list: list, output: str):
    stepwise_features = []
    features = features_list.copy()

    while len(features) > 0:
        mse_dict = {}
        for feature in features:
            feature_list = (stepwise_features + [feature])

            model = LinearRegression(normalize=True)
            model.fit(df[feature_list], df[output])

            predict = model.predict(df_test[feature_list])

            test_output = np.array(df_test[output])

            mse_dict[feature] = mse_error(test_output, predict)

        best_feature = min(mse_dict.keys(), key=(lambda x: mse_dict[x]))

        stepwise_features.append(best_feature)
        features.remove(best_feature)

    return stepwise_features


def backward_selected(df, df_test, features_list: list, output: str):
    last_features = features_list.copy()
    low_priority_features = []
    while len(last_features) > 1:
        mse_dict = {}

        for feature in last_features:
            feature_list = [x for x in last_features if x != feature]

            model = LinearRegression(normalize=True)
            model.fit(df[feature_list], df[output])

            predict = model.predict(df_test[feature_list])

            mse_dict[feature] = mse_error(df_test[output], predict)

        feature_to_remove = min(mse_dict.keys(), key=(lambda x: mse_dict[x]))

        last_features.remove(feature_to_remove)
        low_priority_features.append(feature_to_remove)

    # low_priority_features.reverse()
    return low_priority_features
