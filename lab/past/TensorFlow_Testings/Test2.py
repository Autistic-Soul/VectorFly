#!/usr/bin/env Python
# -*- coding: utf-8 -*-

# TensorFlow, Test: 2
# Auto Encoder

import tensorflow as tf
import numpy as np
import scipy as sp

BATCH_SIZE = 30

class Denoising_AutoEncoder:

    def __init__(self, m_hidden_layers, inputX, corruption_lv = .3):
        self.Weights = None                                                                                             # 输入层到隐含层的 权重矢量/权重向量
        self.Bias = None                                                                                                # 输入层到隐含层的 偏置
        self.Hidden_Outputs = None                                                                                      # 隐含层分层输出
        self.m_hidden_layers = m_hidden_layers                                                                          # 隐含层层数
        self.keep_prob = 1 - corruption_lv                                                                              # 特征保持不变的比例
        self.Weights_Value = None                                                                                       # 权重值
        self.Bias_Value = None                                                                                          # 偏置值
        return

    def fit(self, X, batch_size = BATCH_SIZE):

        self.inputX = X                                                                                                 # 输入数据
        self.m_input_nodes = self.inputX.shape[1]                                                                       # 输入层节点个数

        X_once = tf.placeholder(dtype = "float", shape = [None, self.m_input_nodes], name = "X_once")                   # 将一张图片矢量化表示/向量化表示
        mask = tf.placeholder(dtype = "float", shape = [None, self.m_input_nodes], name = "mask")                       # 用于将部分输入数据置零
        Init_Weights_Absolute_Maximum = 4 * np.sqrt( 6. / (self.m_input_nodes+self.m_hidden_layers) )                   # 权重最大绝对值初始化
        Init_Weights = tf.random_uniform(
            shape = [ self.m_input_nodes, self.m_hidden_layers ],
            minval = -Init_Weights_Absolute_Maximum,
            maxval = Init_Weights_Absolute_Maximum
            )                                                                                                           # 初始化权重

        self.Weights = tf.Variable(initial_value = Init_Weights, name = "Weights")                                      # 编码器 权重
        self.Bias = tf.Variable(initial_value = tf.zeros(shape = [self.m_hidden_layers]), name = "Bias")                # 编码器 偏置
        Decoder_Weights = tf.transpose(x = self.Weights)                                                                # 解码器 权重
        Decoder_Bias = tf.Variable(initial_value = tf.zeros(shape = [self.m_hidden_layers], name = "Decoder_Bias"))     # 解码器 偏置

        X_noised = mask * X_once                                                                                        # 对输入数据加入噪声
        Y = tf.nn.sigmoid(x = tf.matmul(a = X_noised, b = self.Weights) + self.Bias)                                    # 隐含层输出
        Z = tf.nn.sigmoid(x = tf.matmul(a = Y, b = Decoder_Weights) + self.Bias)                                        # 重构输出

        mse = tf.reduce_mean(tf.pow(x = X_noised - Z, y = 2))                                                           # 均方误差(Mean Squared Error)定义

        train_op = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(loss = mse)                         # 最小化均方误差

        X_train = self.inputX

        with tf.Session() as sess:                                                                                      # 新建会话

            tf.initialize_all_variables().run()                                                                         # 初始化全变量

            for i_batch in range(batch_size):

                for start, end in zip( range(0,len(X_train),128), range(128,len(X_train)+128,128) ):
                    _input = X_train[start, end]                                                                        # 设置输入
                    mask_np = np.random.binomial(n = 1, p = self.keep_prob, size = _input.shape)                        # 设置mask
                    sess.run(fetches = train_op, feed_dict = { X_once:_input, mask:mask_np })                           # 开始训练

                if i_batch % 5 == 0:
                    mask_np = np.random.binomial(n = 1, p = 1, size = X_train.shape)
                    print( "batch[%d]:\t\tloss==%s" % (i_batch, sess.run(fetches = mse, feed_dict = { X_once:X_train, mask:mask_np })) )

                self.Weights_Value = self.Weights.eval()                                                                # 保存参数
                self.Bias_Value = self.Bias.eval()
                mask_np = np.random.binomial(n = 1, p = 1, size = X_train.shape)
                self.Hidden_Outputs = Y.eval({ X_once:X_train, mask:mask_np })

    def get_param(self):
        return {
            "Weights": self.Weights_Value,
            "Bias": self.Bias_Value,
            "HiddenOuts": self.Hidden_Outputs
            }
