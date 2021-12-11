import matplotlib.pyplot as plt
import json
import os
import math
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

"""
# 内存中训练
model.train(
    x,
    y,
    epochs=configs['training']['epochs'],
    batch_size=configs['training']['batch_size'],
    save_dir=configs['model']['save_dir']
)
"""
# 记忆外生成训练
steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
model.train_generator(
    data_gen=data.generate_train_batch(
        seq_len=configs['data']['sequence_length'],
        batch_size=configs['training']['batch_size'],
        normalise=configs['data']['normalise']
    ),
    epochs=configs['training']['epochs'],
    batch_size=configs['training']['batch_size'],
    save_dir=configs['model']['save_dir'],
    steps_per_epoch=steps_per_epoch
)

x_test, y_test = data.get_test_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)
# 预测
predictions_multiseq = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'],
                                                        configs['data']['sequence_length'])
predictions_fullseq = model.predict_sequence_full(x_test, configs['data']['sequence_length'])
predictions_pointbypoint = model.predict_point_by_point(x_test)

# 画图
plot_results_multiple(predictions_multiseq, y_test, configs['data']['sequence_length'])
plot_results(predictions_fullseq, y_test)
plot_results(predictions_pointbypoint, y_test)
