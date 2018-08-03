'''
Basic GAN
'''

import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from image_helper import ImageHelper


class BasicGAN:
    
    def __init__(self, hparams):
        self.hparams = hparams
        self._build()

    def _build(self):
        self.noise_dim = self.hparams.noise_dim
        self.learning_rate = self.hparams.learning_rate
        self.epoches = self.hparams.epoches
        self.batch_size = self.hparams.batch_size
        
        self.generator = Generator(self.hparams)
        self.discriminator = Discriminator(self.hparams)
        
        self.rand_noises = tf.placeholder(dtype=tf.float32, shape=[None, self.noise_dim], name="rand_noises")
        self.real_imgs = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 3], name="real_imgs")
        self.fake_imgs = self.generator.generate(self.rand_noises)
        self.fake_logits = self.discriminator.discriminate(self.fake_imgs)
        self.real_logits = self.discriminator.discriminate(self.real_imgs, reuse=True)

        self.d_accuarcy = (tf.reduce_mean(tf.cast(self.real_logits>0, tf.float32)) + 
                           tf.reduce_mean(tf.cast(self.fake_logits<0, tf.float32))) / 2

        with tf.variable_scope("loss"):
            # basic gan
            '''
            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real_logits), 
                                                        logits=self.real_logits))
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.fake_logits), 
                                                        logits=self.fake_logits))
            self.d_loss = self.d_loss_fake + self.d_loss_real
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.fake_logits),
                                                        logits=self.fake_logits))
            '''
            # lease square gan
            self.d_loss_real = tf.reduce_mean(tf.square(self.real_logits - 1))
            self.d_loss_fake = tf.reduce_mean(tf.square(self.fake_logits + 1))
            self.d_loss = self.d_loss_fake + self.d_loss_real
            self.g_loss = tf.reduce_mean(tf.square(self.fake_logits))

        tvars = tf.trainable_variables()
        self.d_vars = [var for var in tvars if 'discriminator' in var.name]
        self.g_vars = [var for var in tvars if 'generator' in var.name]
    
    
    def train(self):
        self.global_step = tf.Variable(0, trainable=False)
        d_optim = tf.train.AdamOptimizer(self.learning_rate).\
                    minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
        g_optim = tf.train.AdamOptimizer(self.learning_rate).\
                    minimize(self.g_loss, var_list=self.g_vars, global_step=self.global_step)

        image_helper = ImageHelper()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for num_epoch, num_batch, batch_images in image_helper.iter_images(
                                                        batch_size=self.batch_size, 
                                                        epoches=self.epoches):
                if (num_epoch == 0) and (num_batch <= 0):
                    # pre-train discriminator
                    _, current_step, d_loss, d_accuarcy = sess.run(
                        [d_optim, self.global_step, self.d_loss, self.d_accuarcy], 
                        feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                                   self.real_imgs: batch_images})
                    if current_step % 5 == 0:
                        print("==== pre-train ==== current_step: ", current_step, "d_loss:", d_loss, "d_accuarcy:", d_accuarcy)        
                else:
                    # optimize discriminator
                    _, current_step, d_loss, d_accuarcy = sess.run(
                        [d_optim, self.global_step, self.d_loss, self.d_accuarcy], 
                        feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                                   self.real_imgs: batch_images})
                    # optimize generator
                    if current_step % 1 == 0:
                        _, g_loss = sess.run(
                            [g_optim, self.g_loss], 
                            feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim])})
                        if current_step % 100 == 0:
                            print("step:", current_step, "d_loss:", d_loss, "d_accuarcy:", d_accuarcy, "g_loss:", g_loss)
    
                    if current_step % 1000 == 0:
                        fake_imgs = sess.run(self.fake_imgs, 
                                             feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim])})
                        image_helper.save_imgs(fake_imgs, img_name="results/fake-{}".format(current_step))

