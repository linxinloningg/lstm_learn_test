from core.model import Model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    return plt


class forecast:
    def __init__(self):
        pass

    def get_data(self, dataframe):
        """

        :param dataframe:
        :return:
        """
        # 获取DataFrame中的数据，形式为数组array形式
        values = dataframe.values
        # 确保所有数据为float类型
        values = values.astype('float32')

        # 将数据变为3d
        X = values.reshape((values.shape[0], 1, values.shape[1]))

        return X

    def forecast(self, dataframe, model_file_path):
        """

        :param dataframe:
        :param model_file_path: ../save_models/03122021-230019-e50.h5
        :return:
        """
        model = Model()
        model.load_model(model_file_path)
        X = self.get_data(dataframe)

        inv_y_predict = model.predict_point_by_point(X)

        # 将数据变为1d
        inv_y_predict = inv_y_predict.reshape(len(inv_y_predict))

        return inv_y_predict


class forecast_scaled:
    def __init__(self):
        # 特征的归一化处理
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def get_data_scaled(self, dataframe, dataframe_scaled):
        """

        :param dataframe:
        :param dataframe_scaled:
        :return:
        """

        dataframe_merge = pd.concat([dataframe_scaled, dataframe])

        # 获取DataFrame中的数据，形式为数组array形式
        values = dataframe_merge.values
        # 确保所有数据为float类型
        values = values.astype('float32')

        scaled = self.scaler.fit_transform(values)

        # 将数据变为3d
        X_total = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))

        return X_total

    def forecast(self, dataframe, dataframe_scaled, model_file_path):
        """

        :param dataframe:
        :param dataframe_scaled:
        :param model_file_path:
        :return:
        """
        model = Model()
        model.load_model(model_file_path)

        X = self.get_data_scaled(dataframe, dataframe_scaled)

        y_predict = model.predict_point_by_point(X[-len(dataframe):, :])

        # 将数据变回2d
        X = X.reshape((X.shape[0], X.shape[2]))
        # invert scaling for forecast
        # 将预测结果按比例反归一化
        inv_y_test = np.concatenate((X[-y_predict.shape[0]:, :dataframe.shape[1] - y_predict.shape[1]], y_predict),
                                    axis=1)
        inv_y_test = self.scaler.inverse_transform(inv_y_test)
        inv_y_predict = inv_y_test[:, -1]

        return inv_y_predict


if __name__ == "__main__":
    dataframe_scaled = pd.read_csv('../data/sh600031.csv')  # 读取股票文件
    dataframe_scaled.set_index('date', inplace=True)
    dataframe_scaled.drop(['volume'], axis=1, inplace=True)

    dataframe = dataframe_scaled.iloc[-30:, :]

    """
    # 归一化数据训练出来的模型
    forecast_scaled = forecast_scaled()
    inv_y_predict_1 = forecast_scaled.forecast(dataframe, dataframe_scaled, "../save_models/03122021-230019-e50.h5")
    plt = plot_results(inv_y_predict_1, dataframe_scaled.iloc[-30:, -1:].values)
    plt.savefig("inv_y_predict_1.jpg")
    """

    # 没有归一化数据训练的模型
    forecast = forecast()
    inv_y_predict_2 = forecast.forecast(dataframe, "../save_models/05122021-213444-e50.h5")
    plt = plot_results(inv_y_predict_2, dataframe_scaled.iloc[-30:, -1:].values)
    plt.savefig("inv_y_predict_2.jpg")
