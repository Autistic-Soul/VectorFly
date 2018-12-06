#!/usr/bin/env Python
# -*- coding: utf-8 -*-

# TensorFlow, Test: 1
# MNIST Classification

from __future__ import absolute_import, \
                       division, \
                       print_function
import argparse, sys
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import tensorflow as tf
FLAGS = None
BATCH_SIZE = 20000
BATCH_SIZE = 50


# 卷积函数, 便于使用卷积算子
def conv2d(x, W):
    return tf.nn.conv2d( input = x, filter = W, strides = [ 1, 1, 1, 1 ], padding = "SAME" )

# 池化函数, 便于使用池化算子
def max_pool(x):
    return tf.nn.max_pool( value = x, ksize = [ 1, 2, 2, 1 ], strides = [ 1, 2, 2, 1 ], padding = "SAME" )

# 权重参数初始化
def weight_variable(shape):
    return tf.Variable( initial_value = tf.truncated_normal( shape = shape, stddev = 0.1 ) )

# 偏置参数初始化(0.1)
def bias_variable(shape):
    return tf.Variable( initial_value = tf.constant( value = 0.1, shape = shape ) )

# 构建网络
def deepNN(x):

    # 输入的数据为扁平的784个数字, 首先将其变成[ batch_num, height, width, channel ]的张量形式
    with tf.name_scope("reshape"):
        x_image = tf.reshape( tensor = x, shape = [ (-1), 28, 28, 1 ] )

    # 第一层卷积, 将1个channel变换到32个
    with tf.name_scope("conv1"):
        W_conv1 = weight_variable([ 5, 5, 1, 32 ])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu( features = ( conv2d( x_image, W_conv1 ) + b_conv1 ) )

    # 第一层池化
    with tf.name_scope("pool1"):
        h_pool1 = max_pool(h_conv1)

    # 第二层卷积, 将32个channel变换到64个
    with tf.name_scope("conv2"):
        W_conv2 = weight_variable([ 5, 5, 32, 64 ])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu( features = ( conv2d( h_pool1, W_conv2 ) + b_conv2 ) )

    # 第二层池化
    with tf.name_scope("pool2"):
        h_pool2 = max_pool(h_conv2)

    # 第一层全连接, 将已变成( 7 * 7 * 64 )规模的参数张量映射到1024个特征上
    with tf.name_scope("fc1"):
        W_fc1 = weight_variable([ ( 7 * 7 * 64 ), 1024 ])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape( tensor = h_pool2, shape = [ (-1), ( 7 * 7 * 64 ) ] )
        h_fc1 = tf.nn.relu( features = ( tf.matmul( a = h_pool2_flat, b = W_fc1 ) + b_fc1 ) )

    # 第一层dropout, 防止特征表示过拟合
    with tf.name_scope("dropout"):
        keep_prob = tf.placeholder( dtype = tf.float32 )
        h_fc1_drop = tf.nn.dropout( x = h_fc1, keep_prob = keep_prob )

    # 第二层全连接, 将特征变换成10维, 每一维的数据对应一个数字的logit值
    with tf.name_scope("fc2"):
        W_fc2 = weight_variable([ 1024, 10 ])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul( a = h_fc1_drop, b = W_fc2 )

    return y_conv, keep_prob


def main(_):

    # 读取数据
    mnist = mnist_data.read_data_sets( train_dir = FLAGS.data_dir, one_hot = True )

    # 准备输入项: 图像和标签
    x = tf.placeholder( dtype = tf.float32, shape = [ None, 784 ] )
    y_ = tf.placeholder( dtype = tf.float32, shape = [ None, 10 ] )

    # 通过构建计算图得到模型结果
    y_conv, keep_prob = deepNN(x)

    # 计算模型结果和真实结果的差距: Cross Entropy Loss(交叉熵)
    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels = y_, logits = y_conv )
        cross_entropy = tf.reduce_mean( input_tensor = cross_entropy )

    # 创建优化器, 对交叉熵进行优化, 进行前向计算, 后向计算, 梯度更新
    with tf.name_scope("adam_optimizer"):
        train_step = tf.train.AdamOptimizer( learning_rate = (1e-4) ).minimize( loss = cross_entropy )

    # 比较交叉熵, 计算精确率
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal( x = tf.argmax( input = y_conv, axis = 1 ), y = tf.argmax( input = y_, axis = 1 ) )
        correct_prediction = tf.cast( x = correct_prediction, DstT = tf.float32 )
    accuracy = tf.reduce_mean( input_tensor = correct_prediction )

    # 创建一个新的session
    with tf.Session() as sess:
        # 初始化session
        sess.run(tf.global_variables_initializer())
        for i in range(BATCH_SIZE):
            batch = mnist.train.next_batch(BATCH_SIZE)
            if not ( i + 1 ) % 100:
                train_accuracy = accuracy.eval(
                    feed_dict = {
                        x : batch[0],
                        y_ : batch[-1],
                        keep_prob : (1.0)
                        }
                    )
                print( "Step[%d]|Trainning Accuracy == %g" % ( i, train_accuracy ) )
            train_step.run(
                feed_dict = {
                    x : batch[0],
                    y_ : batch[-1],
                    keep_prob : (0.5)
                    }
                )
        Test_Accuracy = accuracy.eval(
            feed_dict = {
                x : mnist.test.images,
                y_ : mnist.test.labels,
                keep_prob : (1.0)
                }
            )
        print( "Test Accuracy == %g" % Test_Accuracy )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type = str,
        default = "/tmp/TensorFlow/mnist/input_data",
        help = "Directory for storing input data"
        )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run( main = main, argv = ( [sys.argv[0]] + unparsed ) )


# -*- END -*- #
