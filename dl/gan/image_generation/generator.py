'''
Generator
- input: noises (batch_size, noise_dim)
- output: images (batch_size, height, width, channels)
- arthitecture: CNN, DNN
- use deconvolution (transpose convolution) layer in CNN
'''

import math
import numpy as np
import tensorflow as tf


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

        # 64*64 <- 32 * 32 <- 16*16 <- 8*8 <- 4*4
        self.s_h, self.s_w = self.output_height, self.output_width
        self.s_h2, self.s_w2 = int(math.ceil(self.s_h/2)), int(math.ceil(self.s_w/2))
        self.s_h4, self.s_w4 = int(math.ceil(self.s_h2/2)), int(math.ceil(self.s_w2/2))
        self.s_h8, self.s_w8 = int(math.ceil(self.s_h4/2)), int(math.ceil(self.s_w4/2))
        self.s_h16, self.s_w16 = int(math.ceil(self.s_h8/2)), int(math.ceil(self.s_w8/2))

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.dense_weight = tf.get_variable('dense_weight', [self.noise_dim, 64 * 8 * self.s_h16 * self.s_w16],
                                                initializer=tf.contrib.layers.xavier_initializer())
            
            self.dense_bias = tf.get_variable('dense_bias', [64 * 8 * self.s_h16 * self.s_w16],
                                                initializer=tf.zeros_initializer())
            
            self.filter1_weight = tf.get_variable('filter1', [5, 5, 64*4, 64*8],
                                                  initializer=tf.contrib.layers.xavier_initializer())
    
            self.filter2_weight = tf.get_variable('filter2', [5, 5, 64*2, 64*4],
                                                  initializer=tf.contrib.layers.xavier_initializer())                                      
    
            self.filter3_weight = tf.get_variable('filter3', [5, 5, 64*1, 64*2],
                                                  initializer=tf.contrib.layers.xavier_initializer())                                    
    
            self.filter4_weight = tf.get_variable('filter4', [5, 5, 3, 64*1],
                                                  initializer=tf.contrib.layers.xavier_initializer())                                  

    
    def generate(self, noises, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            output0_ = tf.reshape(tf.nn.xw_plus_b(noises, self.dense_weight, self.dense_bias), 
                                  [-1, self.s_h16, self.s_w16, 64*8])
            output0 = tf.nn.relu(
                tf.layers.batch_normalization(output0_, axis=-1, name="bn-linear", reuse=False))
            
            # TODO batch normalization
            output1_ = tf.nn.conv2d_transpose(output0, 
                                              filter=self.filter1_weight,
                                              output_shape=[self.batch_size, self.s_h8, self.s_w8, 64*4], 
                                              strides=[1,2,2,1], 
                                              name="deconv1")
            output1 = tf.nn.relu(
                tf.layers.batch_normalization(output1_, axis=-1, name="bn-conv1", reuse=False))
    
            output2_ = tf.nn.conv2d_transpose(output1, 
                                              filter=self.filter2_weight,                       
                                              output_shape=[self.batch_size, self.s_h4, self.s_w4, 64*2], 
                                              strides=[1,2,2,1], 
                                              name="deconv2")
            output2 = tf.nn.relu(
                tf.layers.batch_normalization(output2_, axis=-1, name="bn-conv2", reuse=False))
    
            output3_ = tf.nn.conv2d_transpose(output2, 
                                              filter=self.filter3_weight,
                                              output_shape=[self.batch_size, self.s_h2, self.s_w2, 64*1], 
                                              strides=[1,2,2,1], 
                                              name="deconv3")
            output3 = tf.nn.relu(
                tf.layers.batch_normalization(output3_, axis=-1, name="bn-conv3", reuse=False))
    
            output4_ = tf.nn.conv2d_transpose(output3, 
                                              filter=self.filter4_weight,
                                              output_shape=[self.batch_size, self.s_h, self.s_w, 3], 
                                              strides=[1,2,2,1], 
                                              name="deconv4")
        return tf.nn.tanh(output4_)

