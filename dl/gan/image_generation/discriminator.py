'''
Discriminator
- input: images (batch_size, height, width, channels)
- output: scores (batch_size,)
- arthitecture: CNN, DNN
'''

import numpy as np
import tensorflow as tf


class Discriminator:
    
    def __init__(self, name="discriminator", reuse=False):
        self.name = name
        self.reuse = reuse
        self._build()

    def _build(self):
        self.output_height = 64
        self.output_width = 64
        self.batch_size = 25

        with tf.variable_scope(self.name, reuse=self.reuse):
            self.filter1_weight = tf.get_variable(name="filter1", shape=[5, 5, 3, 64], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.filter2_weight = tf.get_variable(name="filter2", shape=[5, 5, 64, 128], 
                                                  initializer=tf.contrib.layers.xavier_initializer())
            self.filter3_weight = tf.get_variable(name="filter3", shape=[5, 5, 128, 128], 
                                                  initializer=tf.contrib.layers.xavier_initializer())            
            self.filter4_weight = tf.get_variable(name="filter4", shape=[5, 5, 128, 128], 
                                                  initializer=tf.contrib.layers.xavier_initializer())

            self.dense0_weight = tf.get_variable(name="dense0_weight", shape=[128*4*4, 128], 
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.dense0_bias = tf.get_variable(name="dense0_bias", shape=[128],
                                               initializer=tf.zeros_initializer())

            self.dense1_weight = tf.get_variable(name="dense1_weight", shape=[128, 1], 
                                                 initializer=tf.contrib.layers.xavier_initializer())
            self.dense1_bias = tf.get_variable(name="dense1_bias", shape=[1],
                                               initializer=tf.zeros_initializer())


    def discriminate(self, inputs):
        # TODO batch normalization
        output1_ = tf.nn.conv2d(input=inputs, filter=self.filter1_weight, strides=[1, 2, 2, 1], 
                                padding="SAME", name="conv1")
        output1 = tf.nn.leaky_relu(output1_)

        output2_ = tf.nn.conv2d(input=output1, filter=self.filter2_weight, strides=[1, 2, 2, 1], 
                                padding="SAME", name="conv2")
        output2 = tf.nn.leaky_relu(output2_)

        output3_ = tf.nn.conv2d(input=output2, filter=self.filter3_weight, strides=[1, 2, 2, 1], 
                                padding="SAME", name="conv3")
        output3 = tf.nn.leaky_relu(output3_)

        output4_ = tf.nn.conv2d(input=output3, filter=self.filter4_weight, strides=[1, 2, 2, 1], 
                                padding="SAME", name="conv4")
        output4 = tf.nn.leaky_relu(output4_)

        flatten_vec = tf.layers.flatten(output4, name="flatten_vec")
        flatten1 = tf.nn.leaky_relu(tf.nn.xw_plus_b(flatten_vec, self.dense0_weight, self.dense0_bias))
        
        return tf.nn.xw_plus_b(flatten1, self.dense1_weight, self.dense1_bias)


if __name__ == "__main__":
    discriminator = Discriminator()
    batch_noise = np.random.random([25, 64, 64, 3]).astype(np.float32)
    pred_score = discriminator.discriminate(batch_noise)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        scores = sess.run(pred_score)
        import IPython
        IPython.embed()
