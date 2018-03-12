# %load model.py
import inspect
import os

import numpy as np
import tensorflow as tf
import time

VGG_MEAN = [103.939, 116.779, 123.68]


class VGG16:
    def __init__(self, vgg16_npy_path, classes=1, shape=(224,224,3)):
        """
        load pre-trained weights from path
        :param vgg16_npy_path: file path of vgg16 pre-trained weights
        """

        # input information
        self.H, self.W, self.C = shape
        self.classes = classes
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self):
        """
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        # input placeholder
        self.x = tf.placeholder(tf.float32, [None, self.H, self.W, self.C])
        self.y = tf.placeholder(tf.float32, [None, self.classes])
        self.w = tf.placeholder(tf.float32, [None, self.classes])
        
        rgb_scaled = self.x

        # normalize input by VGG_MEAN
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert   red.get_shape().as_list()[1:] == [self.H, self.W, 1]
        assert green.get_shape().as_list()[1:] == [self.H, self.W, 1]
        assert  blue.get_shape().as_list()[1:] == [self.H, self.W, 1]

        self.x = tf.concat(axis=3, values=[
              blue - VGG_MEAN[0],
             green - VGG_MEAN[1],
               red - VGG_MEAN[2],
        ])
        assert self.x.get_shape().as_list()[1:] == [self.H, self.W, self.C]

        start_time = time.time()
        print("build model started")

        assert self.x.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(self.x, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1", pretrain=False)
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2", pretrain=False)
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3", pretrain=False)
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1", pretrain=False)
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2", pretrain=False)
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3", pretrain=False)
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.flatten_input = self.flatten(self.pool5)

        self.W1 = tf.get_variable(shape=(self.flatten_input.shape[1],1024), initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="W1", dtype=tf.float32)
        self.b1 = tf.get_variable(shape=1024, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="b1", dtype=tf.float32)
        fc1 = tf.nn.bias_add(tf.matmul(self.flatten_input, self.W1), self.b1)
        self.fc1 = tf.nn.relu(fc1)

        self.W2 = tf.get_variable(shape=(1024,256), initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="W2", dtype=tf.float32)
        self.b2 = tf.get_variable(shape=256, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="b2", dtype=tf.float32)
        fc2 = tf.nn.bias_add(tf.matmul(self.fc1, self.W2), self.b2)
        self.fc2 = tf.nn.relu(fc2)

        self.W3 = tf.get_variable(shape=(256,1), initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="W3", dtype=tf.float32)
        self.b3 = tf.get_variable(shape=1, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="b3", dtype=tf.float32)
        self.output = tf.nn.bias_add(tf.matmul(self.fc2, self.W3), self.b3)

        self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.output, weights=self.w)

        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def flatten(self,bottom):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])
        return x

    def conv_layer(self, bottom, name, pretrain=True):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name, pretrain)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name, pretrain)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def get_conv_filter(self, name, pretrain):
        if pretrain:
            return tf.get_variable(initializer=self.data_dict[name][0], name="filter")
        else:
            return  tf.get_variable(shape=self.data_dict[name][0].shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="filter", dtype=tf.float32) 

    def get_bias(self, name, pretrain):
        if pretrain:
            return tf.get_variable(initializer=self.data_dict[name][1], name="biases")
        else:
            return tf.get_variable(shape=self.data_dict[name][1].shape, initializer=tf.truncated_normal_initializer(mean=0, stddev=0.1), name="biases", dtype=tf.float32) 
