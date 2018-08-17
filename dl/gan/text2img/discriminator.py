'''
Discriminator
- input: images (batch_size, height, width, channels) + tags (batch_size, tag_dim)
- output: scores (batch_size,)
- arthitecture: CNN, DNN
'''

import numpy as np
import tensorflow as tf
from cond_encoder import CondEncoder


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
        self.tag_dim = self.hparams.tag_dim

        self.cond_encoder = CondEncoder(self.hparams)
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.filter1_weight = tf.get_variable(name="filter1", shape=[5, 5, 3, 64], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.filter2_weight = tf.get_variable(name="filter2", shape=[5, 5, 64, 128], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.filter3_weight = tf.get_variable(name="filter3", shape=[5, 5, 128, 256], 
                                                  initializer=tf.contrib.layers.xavier_initializer())            
            self.filter4_weight = tf.get_variable(name="filter4", shape=[5, 5, 256, 512], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.filter5_weight = tf.get_variable(name="filter5", shape=[1, 1, 512+self.tag_dim, 512], 
                                                  initializer=tf.contrib.layers.xavier_initializer())

            self.dense_weight = tf.get_variable(name="dense_weight", shape=[512*4*4, 1], 
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.dense_bias = tf.get_variable(name="dense_bias", shape=[1],
                                               initializer=tf.zeros_initializer())


    def discriminate(self, inputs, tag_oh, reuse=False):
        '''
        @param tag_oh: one hot encoding of the tag
        '''
        with tf.variable_scope(self.name, reuse=reuse):
            output1_ = tf.nn.conv2d(input=inputs, filter=self.filter1_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv1")
            output1 = tf.nn.leaky_relu(output1_)

            output2_ = tf.nn.conv2d(input=output1, filter=self.filter2_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv2")
            output2 = tf.nn.leaky_relu(output2_)
#             output2 = tf.nn.leaky_relu(
#                 tf.layers.batch_normalization(output2_, axis=-1, name="bn-conv2"))

            output3_ = tf.nn.conv2d(input=output2, filter=self.filter3_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv3")
            output3 = tf.nn.leaky_relu(output3_)
#             output3 = tf.nn.leaky_relu(
#                 tf.layers.batch_normalization(output3_, axis=-1, name="bn-conv3"))

            output4_ = tf.nn.conv2d(input=output3, filter=self.filter4_weight, strides=[1, 2, 2, 1], 
                                    padding="SAME", name="conv4")
            output4 = tf.nn.leaky_relu(output4_)
#             output4 = tf.nn.leaky_relu(
#                 tf.layers.batch_normalization(output4_, axis=-1, name="bn-conv4"))

            tag_vec = tf.reshape(self.cond_encoder.get_representation(tag_oh),
                                 [-1, 1, 1, self.tag_dim])
            tiled_tag_vec = tf.tile(tag_vec, [1, 4, 4, 1])
            concat_vec = tf.concat([output4, tiled_tag_vec], axis=-1)

            output5 = tf.nn.conv2d(input=concat_vec, filter=self.filter5_weight, strides=[1, 1, 1, 1], 
                                    padding="VALID", name="conv4")
            flatten_vec = tf.layers.flatten(output5, name="flatten_vec")

        return tf.nn.xw_plus_b(flatten_vec, self.dense_weight, self.dense_bias)

