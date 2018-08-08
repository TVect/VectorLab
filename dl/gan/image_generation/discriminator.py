'''
Discriminator
- input: images (batch_size, height, width, channels)
- output: scores (batch_size,)
- arthitecture: CNN, DNN
'''

import numpy as np
import tensorflow as tf


class Discriminator:
    
    def __init__(self, hparams, name="discriminator", reuse=False):
        self.hparams = hparams
        self.name = name if name else "discriminator"
        self.reuse = reuse
        self._build()

    def _build(self):
        self.output_height = 64
        self.output_width = 64
        self.batch_size = self.hparams.batch_size

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.filter1_weight = tf.get_variable(name="filter1", shape=[4, 4, 3, 64], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.filter2_weight = tf.get_variable(name="filter2", shape=[4, 4, 64, 128], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.filter3_weight = tf.get_variable(name="filter3", shape=[4, 4, 128, 256], 
                                                  initializer=tf.contrib.layers.xavier_initializer())            
            self.filter4_weight = tf.get_variable(name="filter4", shape=[4, 4, 256, 512], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            '''
            self.dense_weight0 = tf.get_variable(name="dense_weight0", shape=[256*4*4, 256], 
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.dense_bias0 = tf.get_variable(name="dense_bias0", shape=[256],
                                               initializer=tf.zeros_initializer())
            '''
            self.dense_weight = tf.get_variable(name="dense_weight", shape=[512*4*4, 1], 
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.dense_bias = tf.get_variable(name="dense_bias", shape=[1],
                                               initializer=tf.zeros_initializer())


    def discriminate(self, inputs, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            output1_ = tf.nn.conv2d(input=inputs, filter=self.filter1_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv1")
            output1 = tf.nn.leaky_relu(output1_)

            output2_ = tf.nn.conv2d(input=output1, filter=self.filter2_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv2")
            if self.hparams.model in ["WGAN", "WGAN-CP"]:
                output2 = tf.nn.leaky_relu(output2_)
            else:
                output2 = tf.nn.leaky_relu(
                    tf.layers.batch_normalization(output2_, axis=-1, name="bn-conv2", reuse=False))
            
            output3_ = tf.nn.conv2d(input=output2, filter=self.filter3_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv3")
            if self.hparams.model in ["WGAN", "WGAN-CP"]: 
                output3 = tf.nn.leaky_relu(output3_)
            else:
                output3 = tf.nn.leaky_relu(
                    tf.layers.batch_normalization(output3_, axis=-1, name="bn-conv3", reuse=False))

            output4_ = tf.nn.conv2d(input=output3, filter=self.filter4_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv4")
            if self.hparams.model in ["WGAN", "WGAN-CP"]:
                output4 = tf.nn.leaky_relu(output4_)
            else:
                output4 = tf.nn.leaky_relu(
                    tf.layers.batch_normalization(output4_, axis=-1, name="bn-conv4", reuse=False))

            flatten_vec = tf.layers.flatten(output4, name="flatten_vec")

            # flatten_vec0 = tf.nn.leaky_relu(tf.nn.xw_plus_b(flatten_vec, self.dense_weight0, self.dense_bias0))

        return tf.nn.xw_plus_b(flatten_vec, self.dense_weight, self.dense_bias)

