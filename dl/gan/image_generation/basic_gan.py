'''
Basic GAN
'''

import os
import numpy as np
import tensorflow as tf
from generator import Generator
from discriminator import Discriminator
from image_helper import ImageHelper
from tensorflow.contrib.training.python.training import hparam


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
        
        self.rand_noises = tf.placeholder(dtype=tf.float32, 
                                          shape=[self.batch_size, self.noise_dim], 
                                          name="rand_noises")
        self.real_imgs = tf.placeholder(dtype=tf.float32, 
                                        shape=[self.batch_size, 64, 64, 3], 
                                        name="real_imgs")
        self.fake_imgs = self.generator.generate(self.rand_noises)
        self.fake_logits = self.discriminator.discriminate(self.fake_imgs)
        self.real_logits = self.discriminator.discriminate(self.real_imgs, reuse=True)

        self.d_accuarcy = (tf.reduce_mean(tf.cast(self.real_logits>0, tf.float32)) + 
                           tf.reduce_mean(tf.cast(self.fake_logits<0, tf.float32))) / 2

        with tf.variable_scope("loss"):
            if self.hparams.model == "GAN":
                # basic gan
                self.d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_logits), 
                                                                   logits=self.real_logits, 
                                                                   label_smoothing=0.2)
                self.d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.fake_logits), 
                                                                   logits=self.fake_logits,
                                                                   label_smoothing=0.2)
                self.d_loss = (self.d_loss_fake + self.d_loss_real) / 2.0
                self.g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_logits),
                                                              logits=self.fake_logits,
                                                              label_smoothing=0.2)
            elif self.hparams.model == "LSGAN":
                # lease square gan
                self.d_loss_real = tf.losses.mean_squared_error(tf.ones_like(self.real_logits), 
                                                                self.real_logits)
                self.d_loss_fake = tf.losses.mean_squared_error(tf.zeros_like(self.fake_logits), 
                                                                self.fake_logits)
                self.d_loss = (self.d_loss_fake + self.d_loss_real) / 2.0
                self.g_loss = tf.losses.mean_squared_error(tf.ones_like(self.fake_logits), self.fake_logits)
            elif self.hparams.model == "WGAN":
                self.d_loss_real = tf.reduce_mean(self.real_logits)
                self.d_loss_fake = tf.reduce_mean(self.fake_logits)
                self.d_loss = self.d_loss_fake - self.d_loss_real
                self.g_loss = -self.d_loss_fake
            elif self.hparams.model == "WGAN-GP":
                '''Gradient Penalty'''
                rand_alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], 
                                               minval=0, maxval=1, name="rand_alpha")
                inter_imgs = self.real_imgs * rand_alpha + self.fake_imgs * (1-rand_alpha) 
                inter_logits = self.discriminator.discriminate(inter_imgs, reuse=True)
                inter_grads = tf.gradients(inter_logits, inter_imgs)[0]
                slops = tf.sqrt(tf.reduce_sum(tf.square(inter_grads), axis=[1,2,3]))
                penalty = tf.reduce_mean(tf.square(slops - 1))

                self.d_loss_real = tf.reduce_mean(self.real_logits)
                self.d_loss_fake = tf.reduce_mean(self.fake_logits)
                self.d_loss = self.d_loss_fake - self.d_loss_real + self.hparams.penalty_coef * penalty
                self.g_loss = -self.d_loss_fake
            else:
                pass

        tvars = tf.trainable_variables()
        self.d_vars = [var for var in tvars if 'discriminator' in var.name]
        self.g_vars = [var for var in tvars if 'generator' in var.name]

        self.global_step = tf.Variable(0, trainable=False)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.hparams.beta1).\
                        minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.hparams.beta1).\
                        minimize(self.g_loss, var_list=self.g_vars)

        if self.hparams.model == "WGAN":
            self.d_optim = tf.train.RMSPropOptimizer(5e-5).\
                            minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
            self.g_optim = tf.train.RMSPropOptimizer(5e-5).\
                            minimize(self.g_loss, var_list=self.g_vars)
            self.clip_min = self.hparams.clip_min
            self.clip_max = self.hparams.clip_max
            with tf.control_dependencies([self.d_optim]):
#                 self.d_optim = tf.group(
#                     *(tf.assign(var, tf.clip_by_value(var, self.clip_min, self.clip_max)) 
#                       for var in tvars if ("discriminator/filter" in var.name) or 
#                       ("discriminator/dense_weight" in var.name)))
                self.d_optim = tf.group(
                    *(tf.assign(var, tf.clip_by_value(var, self.clip_min, self.clip_max)) for var in self.d_vars))

        self._add_saver()


    def _add_saver(self):
        # checkpoint 相关
        self.checkpoint_dir = os.path.abspath(os.path.join(self.hparams.checkpoint_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model_{}".format(self.hparams.model))
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.hparams.max_to_keep)


    def train(self, sess):
        # loss summaries
        d_summary_op = tf.summary.merge([tf.summary.histogram("d_real_prob", tf.sigmoid(self.real_logits)),
                                         tf.summary.histogram("d_fake_prob", tf.sigmoid(self.fake_logits)),
                                         tf.summary.scalar("d_loss_fake", self.d_loss_fake), 
                                         tf.summary.scalar("d_loss_real", self.d_loss_real), 
                                         tf.summary.scalar("d_loss", self.d_loss)],
                                        name="discriminator_summary")
        g_summary_op = tf.summary.merge([tf.summary.histogram("g_prob", tf.sigmoid(self.fake_logits)),
                                         tf.summary.scalar("g_loss", self.g_loss),
                                         tf.summary.image("gen_images", self.fake_imgs)],
                                        name="generator_summary")

        self.summary_dir = os.path.abspath(os.path.join(self.hparams.checkpoint_dir, "summary"))
        summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

        image_helper = ImageHelper()
        
        sess.run(tf.global_variables_initializer())

        for num_epoch, num_batch, batch_images in image_helper.iter_images(
                                                    dirname=self.hparams.data_dir,
                                                    batch_size=self.batch_size, 
                                                    epoches=self.epoches):
            if (num_epoch == 0) and (num_batch < self.hparams.d_pretrain):
                # pre-train discriminator
                _, current_step, d_loss, d_accuarcy = sess.run(
                    [self.d_optim, self.global_step, self.d_loss, self.d_accuarcy], 
                    feed_dict={
                        self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                        self.real_imgs: batch_images})
                if current_step == self.hparams.d_pretrain:
                    tf.logging.info("==== pre-train ==== current_step:{}, d_loss:{}, d_accuarcy:{}"\
                                    .format(current_step, d_loss, d_accuarcy))
            else:
                # optimize discriminator
                _, current_step, d_loss, d_accuarcy = sess.run(
                    [self.d_optim, self.global_step, self.d_loss, self.d_accuarcy], 
                    feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                               self.real_imgs: batch_images})

                # optimize generator
                if current_step % self.hparams.d_schedule == 0:
                    _, g_loss = sess.run(
                        [self.g_optim, self.g_loss], 
                        feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim])})

                # summary
                if current_step % self.hparams.log_interval == 0:
                    d_summary_str, g_summary_str = sess.run(
                        [d_summary_op, g_summary_op], 
                        feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                                   self.real_imgs: batch_images})
                    summary_writer.add_summary(d_summary_str, current_step)
                    summary_writer.add_summary(g_summary_str, current_step)

                    tf.logging.info("step:{}, d_loss:{}, d_accuarcy:{}, g_loss:{}"\
                                    .format(current_step, d_loss, d_accuarcy, g_loss))

            if (num_epoch > 0) and (num_batch == 0):
                # generate images per epoch
                tf.logging.info("epoch:{} === generate images and save checkpoint".format(num_epoch))
                fake_imgs = sess.run(
                    self.fake_imgs, 
                    feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim])})
                image_helper.save_imgs(fake_imgs, 
                                       img_name="{}/fake-{}".format(self.hparams.sample_dir, num_epoch))
                # save model per epoch
                self.saver.save(sess, self.checkpoint_prefix, global_step=num_epoch)


    def infer(self, sess):
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        image_helper = ImageHelper()

        fake_imgs = sess.run(
            self.fake_imgs, 
            feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim])})
        img_name = "{}/infer-image".format(self.hparams.sample_dir)
        image_helper.save_imgs(fake_imgs, 
                               img_name=img_name)

        tf.logging.info("====== generate images in file: {} ======".format(img_name))

