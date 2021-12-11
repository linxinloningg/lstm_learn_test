import matplotlib.pyplot as plt
import json
import os
from core.data_processor import DataLoader
from core.model import Model


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # 填充预测列表，将其在图表中移动到正确的开始位置
    for i, data in enumerate(predicted_data):
        padding = [None for _ in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


configs = json.load(open('config.json', 'r'))

data = DataLoader(
    filename=os.path.join('../data', configs['data']['filename']),
    split=configs['data']['train_test_split'],
    cols=configs['data']['columns']
)

x, y = data.get_train_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)

model = Model()
model.build_model(configs)

model.train(
    x,
    y,
    epochs=configs['training']['epochs'],
    batch_size=configs['training']['batch_size'],
    save_dir=configs['model']['save_dir']
)

x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)

predictions_bypoint = model.predict_point_by_point(x_test)
plot_results(predictions_bypoint, y_test)


predictions_byfull = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
plot_results(predictions_byfull, y_test)

