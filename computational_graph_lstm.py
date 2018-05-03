#-*- coding:utf-8 -*


import tensorflow as tf
from LSTM_captcha.config import *



def computational_graph_lstm(x, y, batch_size = batch_size):

    #weights and biases of appropriate shape to accomplish above task
    out_weights = tf.Variable(tf.random_normal([num_units,n_classes]), name = 'out_weight')
    out_bias = tf.Variable(tf.random_normal([n_classes]),name = 'out_bias')

    #构建网络
    lstm_layer = [tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True) for _ in range(layer_num)]    #创建两层的lstm
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layer, state_is_tuple = True)   #将lstm连接在一起

    init_state = mlstm_cell.zero_state(batch_size, tf.float32)  #cell的初始状态

    outputs = list()    #每个cell的输出
    state = init_state
    with tf.variable_scope('RNN'):
        for timestep in range(time_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = mlstm_cell(x[:, timestep, :], state) # 这里的state保存了每一层 LSTM 的状态
            outputs.append(cell_output)
    # h_state = outputs[-1] #取最后一个cell输出

    #计算输出层的第一个元素
    prediction_1 = tf.nn.softmax(tf.matmul(outputs[-4],out_weights)+out_bias)    #获取最后time-step的输出，使用全连接, 得到第一个验证码输出结果
    #计算输出层的第二个元素
    prediction_2 = tf.nn.softmax(tf.matmul(outputs[-3],out_weights)+out_bias)   #输出第二个验证码预测结果
    #计算输出层的第三个元素
    prediction_3 = tf.nn.softmax(tf.matmul(outputs[-2],out_weights)+out_bias)   #输出第三个验证码预测结果
    #计算输出层的第四个元素
    prediction_4 = tf.nn.softmax(tf.matmul(outputs[-1],out_weights)+out_bias)   #输出第四个验证码预测结果,size:[batch,num_class]
    #输出连接
    prediction_all = tf.concat([prediction_1, prediction_2, prediction_3, prediction_4],1)  # 4 * [batch, num_class] => [batch, 4 * num_class]
    prediction_all = tf.reshape(prediction_all,[batch_size, captcha_num, n_classes],name ='prediction_merge') # [4, batch, num_class] => [batch, 4, num_class]

    #loss_function
    loss = -tf.reduce_mean(y * tf.log(prediction_all),name = 'loss')
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_all,labels=y))
    #optimization
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate, name = 'opt').minimize(loss)
    #model evaluation
    pre_arg = tf.argmax(prediction_all,2,name = 'predict')
    y_arg = tf.argmax(y,2)
    correct_prediction = tf.equal(pre_arg, y_arg)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name = 'accuracy')

    return opt, loss, accuracy, pre_arg, y_arg



