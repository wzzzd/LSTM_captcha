# LSTM_captcha
## 基于tensorflow的LSTM网络识别验证码

### 1、前面工作
关于验证码识别，试过使用传统的machine learning方式识别，在相同样本下效果还算可以，但当迁移到别的数据集时，效果不理想。<br>
对于使用深度学习识别验证码，尝试过使用LeNet-5、AlexNet网络，可能是网络结构简单的原因，结果不收敛。故尝试用了RNN中的LSTM单元网络来识别，效果较理想。

### 2、原始验证码文件
![验证码](https://github.com/wzzzd/LSTM_captcha/blob/master/picture/3AWM.jpg)<br>
![验证码](https://github.com/wzzzd/LSTM_captcha/blob/master/picture/D9XV.jpg)<br>
![验证码](https://github.com/wzzzd/LSTM_captcha/blob/master/picture/ZM19.jpg)

### 3、网络结构
![network structure](https://github.com/wzzzd/LSTM_captcha/blob/master/picture/structure.png)

### 4、训练过程
使用Adam算法替代梯度下降，迭代到3000次，accuracy达0.65，loss小于0.03。继续进行迭代、或优化能到达更高的准确率。
![验证码](https://github.com/wzzzd/LSTM_captcha/blob/master/picture/accuracy.png)<br>
![验证码](https://github.com/wzzzd/LSTM_captcha/blob/master/picture/loss.png)


### 6、总结
this is a placeholder.









