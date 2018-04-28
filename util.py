#-*- coding:utf-8 -*

import os
import random
import numpy as np
from PIL import Image
from captcha_test.captcha_lstm.config import *



def get_batch(data_path = captcha_path, is_training = True):
    target_file_list = os.listdir(data_path)    #读取路径下的所有文件名

    batch = batch_size if is_training else len(target_file_list)   # 确认batch 大小
    batch_x = np.zeros([batch, time_steps, n_input])   #batch 数据
    batch_y = np.zeros([batch, captcha_num, n_classes])   # batch 标签

    for i in range(batch):
        file_name = random.choice(target_file_list) if is_training else target_file_list[i] #确认要打开的文件名
        img = Image.open(data_path + '/' + file_name) #打开图片
        img = np.array(img)
        if len(img.shape) > 2:
            img = np.mean(img, -1)  #转换成灰度图像:(26,80,3) =>(26,80)
            img = img / 255   #标准化，为了防止训练集的方差过大而导致的收敛过慢问题。
            # img = np.reshape(img,[time_steps,n_input])  #转换格式：(2080,) => (26,80)
        batch_x[i] = img

        label = np.zeros(captcha_num * n_classes)
        for num, char in enumerate(file_name.split('.')[0]):
            index = num * n_classes + char2index(char)
            label[index] = 1
        label = np.reshape(label,[captcha_num, n_classes])
        batch_y[i] = label
    return batch_x, batch_y


def char2index(c):
    k = ord(c)
    index = -1
    if k >= 48 and k <= 57: #数字索引
        index = k - 48
    if k >= 65 and k <= 90: #大写字母索引
        index = k - 55
    if k >= 97 and k <= 122: #小写字母索引
        index = k - 61
    if index == -1:
        raise ValueError('No Map')
    return index


def index2char(k):
    # k = chr(num)
    index = -1
    if k >= 0 and k < 10: #数字索引
        index = k + 48
    if k >= 10 and k < 36: #大写字母索引
        index = k + 55
    if k >= 36 and k < 62: #小写字母索引
        index = k + 61
    if index == -1:
        raise ValueError('No Map')
    return chr(index)

# print(index2char(61))


