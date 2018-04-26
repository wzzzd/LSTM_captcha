#-*- coding:utf-8 -*

import os
import random
import numpy as np
from PIL import Image
from captcha_test.captcha_soc_dl.config import *



#打开图片
def open_image_file(path):
    img = Image.open(path)
    return img


def get_batch():
    target_file_list = os.listdir(captcha_path)

    batch_x = np.zeros([batch_size, image_height, image_width, image_channels])
    batch_y = np.zeros([batch_size, captcha_num * char_set_len])

    for i in range(batch_size):
        file_name = random.choice(target_file_list) #随机选择某个文件名
        img = Image.open(captcha_path + '/' + file_name) #打开图片
        img = np.array(img)
        # print(np.array(img).shape)
        if len(img.shape) > 2:
            img = np.mean(img, -1)  #转换成灰度图像:(26,80,3) =>(26,80)
            img = img.flatten() / 255   #标准化，为了防止训练集的方差过大而导致的收敛过慢问题。
            # print('before:',img)
            img = np.reshape(img,[image_height,image_width,image_channels])  #转换格式：(26,80) =>(26,80,1)
            # print('affter:',img)
        batch_x[i] = img

        label = np.zeros(captcha_num * char_set_len)
        for num, char in enumerate(file_name.split('.')[0]):
            index = num * char_set_len + char2index(char)
            label[index] = 1
        batch_y[i] = label
        # print(label)
    # print(batch_x.shape)
    # print(batch_y.shape)
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


# get_batch()

