'''
Generator
- input: noises (batch_size, noise_dim) + tags (batch_size, tag_dim)
- output: images (batch_size, height, width, channels)
- arthitecture: CNN, DNN
- use deconvolution (transpose convolution) layer in CNN
'''

import math
import numpy as np
import tensorflow as tf
from cond_encoder import CondEncoder


class Generator:
    
    def __init__(self, hparams, name="generator", reuse=False):
        self.hparams = hparams
        self.name = name if name else "generator"
        self.reuse = reuse
        self._build()
    
    
    def _build(self):
        self.output_height = 64
        self.output_width = 64
        self.noise_dim = self.hparams.noise_dim
        self.batch_size = self.hparams.batch_size
        self.tag_dim = self.hparams.tag_dim

        self.cond_encoder = CondEncoder(self.hparams)
        # 64*64 <- 32 * 32 <- 16*16 <- 8*8 <- 4*4
        with tf.variable_scope(self.name, reuse=self.reuse):
            self.dense_weight = tf.get_variable('dense_weight', 
                                                [self.noise_dim+self.tag_dim, 64*8*4*4],
                                                initializer=tf.contrib.layers.xavier_initializer())
            
            self.dense_bias = tf.get_variable('dense_bias', [64*8*4*4],
                                                initializer=tf.zeros_initializer())
            
            self.filter1_weight = tf.get_variable('filter1', [5, 5, 64*4, 64*8],
                                                  initializer=tf.contrib.layers.xavier_initializer())
    
            self.filter2_weight = tf.get_variable('filter2', [5, 5, 64*2, 64*4],
                                                  initializer=tf.contrib.layers.xavier_initializer())                                      
    
            self.filter3_weight = tf.get_variable('filter3', [5, 5, 64*1, 64*2],
                                                  initializer=tf.contrib.layers.xavier_initializer())                                    
    
            self.filter4_weight = tf.get_variable('filter4', [5, 5, 3, 64*1],
                                                  initializer=tf.contrib.layers.xavier_initializer())                                  


    def generate(self, noises, tag_oh, reuse=False):
        '''
        @param tag_oh: one hot encoding of the tag
        '''
        with tf.variable_scope(self.name, reuse=reuse):
            tag_vec = self.cond_encoder.get_representation(tag_oh)

            output0_ = tf.reshape(
                tf.nn.xw_plus_b(tf.concat([noises, tag_vec], axis=-1), self.dense_weight, self.dense_bias), 
                [-1, 4, 4, 512])
            output0 = tf.nn.relu(
                tf.layers.batch_normalization(output0_, axis=-1, name="bn-linear"))

            # TODO batch normalization
            output1_ = tf.nn.conv2d_transpose(output0, 
                                              filter=self.filter1_weight,
                                              output_shape=[self.batch_size, 8, 8, 64*4], 
                                              strides=[1,2,2,1], 
                                              name="deconv1")
            output1 = tf.nn.relu(
                tf.layers.batch_normalization(output1_, axis=-1, name="bn-conv1"))

            output2_ = tf.nn.conv2d_transpose(output1, 
                                              filter=self.filter2_weight,                       
                                              output_shape=[self.batch_size, 16, 16, 64*2], 
                                              strides=[1,2,2,1], 
                                              name="deconv2")
            output2 = tf.nn.relu(
                tf.layers.batch_normalization(output2_, axis=-1, name="bn-conv2"))
    
            output3_ = tf.nn.conv2d_transpose(output2, 
                                              filter=self.filter3_weight,
                                              output_shape=[self.batch_size, 32, 32, 64*1], 
                                              strides=[1,2,2,1], 
                                              name="deconv3")
            output3 = tf.nn.relu(
                tf.layers.batch_normalization(output3_, axis=-1, name="bn-conv3"))
    
            output4_ = tf.nn.conv2d_transpose(output3, 
                                              filter=self.filter4_weight,
                                              output_shape=[self.batch_size, 64, 64, 3], 
                                              strides=[1,2,2,1], 
                                              name="deconv4")
        return tf.nn.tanh(output4_)
