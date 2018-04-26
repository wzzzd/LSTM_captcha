#-*- coding:utf-8 -*

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import rnn
from captcha_test.captcha_lstm.computational_graph_lstm import *



def train():

    # defining placeholders
    x = tf.placeholder("float",[None,time_steps,n_input]) #input image placeholder
    y = tf.placeholder("float",[None,captcha_num,n_classes])  #input label placeholder

    # computational graph
    opt, loss, accuracy, pre_arg, y_arg = computational_graph_lstm(x, y)

    saver = tf.train.Saver()  # 创建训练模型保存类
    init = tf.global_variables_initializer()    #初始化变量值

    with tf.Session() as sess:  # 创建tensorflow session
        sess.run(init)
        iter = 1
        while True:
            batch_x, batch_y = get_batch()
            batch_y = batch_y.reshape((batch_size, captcha_num, n_classes))
            batch_x = batch_x.reshape((batch_size,time_steps,n_input))    #转换格式
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})   #只运行优化迭代计算图
            if iter %100==0:
                los, acc, parg, yarg = sess.run([loss, accuracy, pre_arg, y_arg],feed_dict={x:batch_x,y:batch_y})
                print("For iter ",iter)
                print("Accuracy ",acc)
                print("Loss ",los)
                if iter % 1000 ==0:
                    print("predict arg:",parg[0:10])
                    print("yarg :",yarg[0:10])
                print("__________________")
                if acc > 0.50:
                    print("training complete, accuracy:", acc)
                    break
            if iter % 1000 == 0:   #保存模型
                saver.save(sess, model_path, global_step=iter)
            iter += 1
        # 计算测试准确率
        test_data, test_label = get_batch()
        test_label = test_label.reshape((-1, captcha_num, char_set_len))
        test_data = test_data.reshape((-1,time_steps,n_input))    #转换格式
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


# def save_mark(iter_mark, acc, loss):
#     with open("C:/Users/Administrator/PycharmProjects/NeuralNetwork/mnist/result/result.txt",'a') as f:
#         for num, ite in enumerate(iter_mark):
#             f.write(str(ite)+'\t'+str(acc[num])+'\t'+str(loss[num])+'\n')
# save_mark(iter_mark,acc_mark,loss_mark)

if __name__ == '__main__':
    train()

