import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class OutlierDetectHelper:
    __df: pd.DataFrame
    __desc_sorted_indices = None
    __desc_sorted_indices_value: bool = False  # indicate to  __desc_sorted_indices has any value or not!
    __x: str
    __y: str

    def outlier_selection_by_linear_model(self, df, x: str, y: str):
        self.__df = df
        self.__x = x
        self.__y = y

        # get numpy data:
        x = self.__df[x].values
        x = np.reshape(x, [-1, 1])
        y = self.__df[y].values
        y = np.reshape(y, [-1, 1])

        from sklearn.linear_model import Ridge
        model = Ridge(0)  # least square
        model.fit(x, y)
        pred = model.predict(x)

        # Find the top most data point with highest error:
        error = np.abs(y - pred)  # get diff of real output and predicted output

        #   sort the indices in descending order:
        sorted_error = np.argsort(error, axis=0)
        sorted_error_reversed = np.flip(sorted_error)

        self.__desc_sorted_indices = sorted_error_reversed
        self.__desc_sorted_indices_value = True  # there are some indices now
        return

    def plot_excluded_outliers(self, xlim=None, ylim=None, k: int = 10):

        if self.__desc_sorted_indices_value is False:  # if desc_sorted_indices is empty
            raise Exception(
                'There is no any indices to plot, call the outlier_selection_by_linear_model function first')

        k_indices = self.__desc_sorted_indices[0:k]  # select k data point

        # make a vector of str colors with 'r' and 'b' color, blue for included data point
        #   and red for excluded data points:
        col = []
        for i in range(self.__df.shape[0]):
            col.insert(i, 'b')
            for j in range(k_indices.shape[0]):
                if i == k_indices[j]:
                    col[i] = 'r'
                    break

        # visualization the data points:
        plt.scatter(self.__df[self.__x], self.__df[self.__y], marker='.', c=col)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.show()
        return

    def get_k_outlier(self, k):
        return self.__desc_sorted_indices[0:k]  # select k data point


