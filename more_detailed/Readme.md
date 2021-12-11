# LSTM对股票的收益进行预测（Sequential 序贯模型，Keras实现）更详细步骤和分析

#### 总包含文章：

* [一个完整的机器学习模型的流程](https://blog.csdn.net/linxinloningg/article/details/121685647)
* [浅谈深度学习：了解RNN和构建并预测](https://blog.csdn.net/linxinloningg/article/details/121881042)
* [浅谈深度学习：基于对LSTM项目`LSTM Neural Network for Time Series Prediction`的理解与回顾](https://blog.csdn.net/linxinloningg/article/details/121881068)
* [浅谈深度学习：LSTM对股票的收益进行预测（Sequential 序贯模型，Keras实现）](https://blog.csdn.net/linxinloningg/article/details/121881117)

### Sequential 序贯模型

　　序贯模型是函数式模型的简略版，为最简单的线性、从头到尾的结构顺序，不分叉，是多个网络层的线性堆叠。

### 编译

　　在训练模型之前，我们需要配置学习过程，这是通过compile方法完成的，他接收三个参数：

    * 优化器 optimizer：它可以是现有优化器的字符串标识符，如 rmsprop 或 adagrad，也可以是 Optimizer 类的实例。详见：optimizers。
    
    * 损失函数 loss：模型试图最小化的目标函数。它可以是现有损失函数的字符串标识符，如 categorical_crossentropy 或 mse，也可以是一个目标函数。详见：losses。
    
    * 评估标准 metrics：对于任何分类问题，你都希望将其设置为 metrics = ['accuracy']。评估标准可以是现有的标准的字符串标识符，也可以是自定义的评估标准函数。

### 训练

　　Keras 模型在输入数据和标签的 Numpy 矩阵上进行训练。为了训练一个模型，你通常会使用 fit 函数。

### 参数详解：
* x=None, #输入的x值
* y=None, #输入的y标签值
* batch_size=None, #整数 ，每次梯度更新的样本数即批量大小。未指定，默认为32。
* epochs=1, #迭代次数
* verbose=1, #整数，代表以什么形式来展示日志状态，
* verbose = 0 为不在标准输出流输出日志信息，verbose = 1 为输出进度条记录，verbose = 2 为每个epoch输出一行记录
* callbacks=None, #回调函数，这个list中的回调函数将会在训练过程中的适当时机被调用，参考回调函数
* validation_split=0.0, #浮点数0-1之间，用作验证集的训练数据的比例。模型将分出一部分不会被训练的验证数据，并将在每一轮结束时评估这些验证数据的误差和任何其他模型指标。
* validation_data=None, #这个参数会覆盖 validation_split 即两个函数只能存在一个，它的输入为元组 (x_val，y_val)，这作为验证数据。
* shuffle=True, #布尔值。是否在每轮迭代之前混洗数据
* class_weight=None,
* sample_weight=None,
* initial_epoch=0,
* steps_per_epoch=None, #一个epoch包含的步数（每一步是一个batch的数据送入） 当使用如TensorFlow数据Tensor之类的输入张量进行训练时，默认的None代表自动分割，即数据集样本数/batch样本数。
* validation_steps=None, #在验证集上的step总数，仅当steps_per_epoch被指定时有用。
* validation_freq=1, #指使用验证集实施验证的频率。当等于1时代表每个epoch结束都验证一次
* max_queue_size=10,
* workers=1,
* use_multiprocessing=False

### 基于栈式 LSTM 的序列分类

在这个模型中，我们将尝试将多个LSTM 层叠在一起，使模型能够学习更高层次的时间表示。

步骤例子如下：

前两个 LSTM 返回完整的输出序列，但最后一个只返回输出序列的最后一步，从而降低了时间维度（即将输入序列转换成单个向量）。

![stacked LSTM](Readme.assets/regular_stacked_lstm.png)

添加层：

* 

  * LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq)

  * Dropout(dropout_rate)

  * Dense(neurons, activation=activation)

* test_1:

  * model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='relu'))

* test_2:
  * model.add(SimpleRNN(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(100))
    model.add(Dropout(0.2))
    model.add(Dense(1,activation='relu'))

* test_3:

  * model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(100,return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(1,activation='relu'))
  
* test_4

  * model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))

    model.add(Dropout(0.2))

    model.add(LSTM(100,return_sequences=True))
    model.add(LSTM(100,return_sequences=False))

    model.add(Dropout(0.2))

    model.add(Dense(1,activation='relu'))