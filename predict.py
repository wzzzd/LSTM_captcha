#-*- coding:utf-8 -*

import tensorflow as tf
from LSTM_captcha.computational_graph_lstm import *
from LSTM_captcha.config import *
from LSTM_captcha.util import *

# def get_test_set():
#
#     target_file_list = os.listdir(validation_path)   #获取测试集路径下的所有文件
#     print("预测的验证码文件:",len(target_file_list))
#
#     #判断条件
#     flag = len(target_file_list) // batch_size  #计算待检测验证码个数能被batch size 整除的次数
#     batch_len = flag if flag > 0 else 1  #共有多少个batch
#     flag2 = len(target_file_list) % batch_size  #计算验证码被batch size整除后的取余
#     batch_len = batch_len if flag2 == 0 else batch_len + 1  #若不能整除，则batch数量加1
#
#     print("共生成batch数:",batch_len)
#     print("验证码根据batch取余:",flag2)
#
#     batch_data = []
#     batch_file_name = []
#     k = 0
#     for i in range(batch_len):
#         batch_temp =  np.zeros([batch_size, time_steps, n_input])
#         batch_temp_file = []
#         for j in range(batch_size):
#             if i == batch_len -1 and j >= flag2:
#                 batch_temp_file.append("0")
#                 continue
#             batch_temp[j] = open_iamge(target_file_list[k])
#             batch_temp_file.append(target_file_list[k])
#             k += 1
#         batch_data.append(batch_temp)
#         batch_file_name.append(batch_temp_file)
#     # print("batch data size:",np.array(batch_data).shape)
#     # print("batch_file_name size:",np.array(batch_file_name).shape)
#     print("data:",batch_data)
#     print("file name:",batch_file_name)
#     return batch_data, target_file_list #batch_file_name

def get_test_set():

    target_file_list = os.listdir(test_data_path)   #获取测试集路径下的所有文件
    print("预测的验证码文件:",len(target_file_list))

    #判断条件
    flag = len(target_file_list) // batch_size  #计算待检测验证码个数能被batch size 整除的次数
    batch_len = flag if flag > 0 else 1  #共有多少个batch
    flag2 = len(target_file_list) % batch_size  #计算验证码被batch size整除后的取余
    batch_len = batch_len if flag2 == 0 else batch_len + 1  #若不能整除，则batch数量加1

    print("共生成batch数:",batch_len)
    print("验证码根据batch取余:",flag2)

    batch =  np.zeros([batch_len * batch_size, time_steps, n_input])
    for i, file in enumerate(target_file_list):
        batch[i] = open_iamge(file)
    batch = batch.reshape([batch_len, batch_size, time_steps, n_input])
    return batch, target_file_list #batch_file_name



def open_iamge(file):
    img = Image.open(test_data_path + '/' + file) #打开图片
    img = np.array(img)
    if len(img.shape) > 2:
        img = np.mean(img, -1)  #转换成灰度图像:(26,80,3) =>(26,80)
        img = img / 255
    return img



def predict():

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(path + "/model/" + "model.ckpt-5000.meta")
        saver.restore(sess, tf.train.latest_checkpoint(path + "/model/")) #读取已训练模型

        graph = tf.get_default_graph()  #获取原始计算图，并读取其中的tensor
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        pre_arg = graph.get_tensor_by_name("predict:0")

        test_x, file_list = get_test_set()  #获取测试集
        predict_result = []
        for i in range(len(test_x)):
            batch_test_x = test_x[i]
            batch_test_y = np.zeros([batch_size, captcha_num,n_classes])    #创建空的y输入
            test_predict = sess.run([pre_arg], feed_dict={x: batch_test_x, y:batch_test_y})
            # print(test_predict)
            # predict_result.extend(test_predict)

            for line in test_predict[0]:    #将预测结果转换为字符
                character = ""
                for each in line:
                    character += index2char(each)
                predict_result.append(character)

        predict_result = predict_result[:len(file_list)]    #预测结果
        write_to_file(predict_result, file_list)    #保存到文件


def write_to_file(predict_list, file_list):
    with open(output_path, 'a') as f:
        for i, res in enumerate(predict_list):
            if i == 0:
                f.write("id\tfile\tresult\n")
            f.write(str(i) + "\t" + file_list[i] + "\t" + res + "\n")
    print("预测结果保存在：",output_path)


if __name__ == '__main__':
    predict()
    # get_test_set()
