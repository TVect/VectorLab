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

        self._add_placeholder()
        self._add_loss()
        self._add_optim()
        self._add_saver()


    def _add_placeholder(self):
        self.rand_noises = tf.placeholder(tf.float32, [self.batch_size, self.noise_dim], "rand_noises")
        self.real_imgs = tf.placeholder(tf.float32, [self.batch_size, 64, 64, 3], "real_imgs")
        self.tags = tf.placeholder(tf.float32, [self.batch_size, 22], "tags")
        self.wrong_tags = tf.placeholder(tf.float32, [self.batch_size, 22], "wrong_tags") 


    def _add_loss(self):
        images_filped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), self.real_imgs)
        angles = tf.random_uniform([self.batch_size], 
                                   minval=-15.0 * np.pi / 180.0, 
                                   maxval=15.0 * np.pi / 180.0)
        self.rotated_imgs = tf.contrib.image.rotate(images_filped, angles, interpolation='NEAREST')

        self.fake_imgs = self.generator.generate(self.rand_noises, self.tags)
        # fake images, right tag 的 logits
        self.fake_logits = self.discriminator.discriminate(self.fake_imgs, self.tags)
        # real images, wrong tag 的 logits !!! 新增的一种 loss
        self.wtag_logits = self.discriminator.discriminate(self.rotated_imgs, self.wrong_tags, reuse=True)
        # real images, right tag 的 logits
        self.real_logits = self.discriminator.discriminate(self.real_imgs, self.tags, reuse=True)

        self.d_accuarcy = (tf.reduce_mean(tf.cast(self.real_logits>0, tf.float32)) + 
                           tf.reduce_mean(tf.cast(self.fake_logits<0, tf.float32)) + 
                           tf.reduce_mean(tf.cast(self.wtag_logits<0, tf.float32))) / 3.0

#         self.d_accuarcy = (tf.reduce_mean(tf.cast(self.real_logits>0, tf.float32)) + 
#                            tf.reduce_mean(tf.cast(self.fake_logits<0, tf.float32))) / 2.0

        '''
        # basic gan
        self.d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_logits), 
                                                           logits=self.real_logits, 
                                                           label_smoothing=0.2)
        self.d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.fake_logits), 
                                                           logits=self.fake_logits,
                                                           label_smoothing=0.2)
        self.d_loss_wtag = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.wtag_logits), 
                                                           logits=self.wtag_logits, 
                                                           label_smoothing=0.2)
        self.d_loss = self.d_loss_real + (self.d_loss_fake + self.d_loss_wtag) / 2.0
        # self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_logits),
                                                      logits=self.fake_logits,
                                                      label_smoothing=0.2)
        '''
        # Gradient Penalty
        rand_alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], 
                                       minval=0, maxval=1, name="rand_alpha")
        inter_imgs = self.real_imgs * rand_alpha + self.fake_imgs * (1-rand_alpha) 
        inter_logits = self.discriminator.discriminate(inter_imgs, self.tags, reuse=True)
        inter_grads = tf.gradients(inter_logits, inter_imgs)[0]
        slops = tf.sqrt(tf.reduce_sum(tf.square(inter_grads), axis=[1,2,3]))
        penalty = tf.reduce_mean(tf.square(slops - 1))

        self.d_loss_real = tf.reduce_mean(self.real_logits)
        self.d_loss_fake = tf.reduce_mean(self.fake_logits)
        self.d_loss_wtag = tf.reduce_mean(self.wtag_logits)
        self.d_loss = (self.d_loss_fake + self.d_loss_wtag) - self.d_loss_real + self.hparams.penalty_coef * penalty
        # self.d_loss = self.d_loss_fake - self.d_loss_real + self.hparams.penalty_coef * penalty
        self.g_loss = -self.d_loss_fake

        # self.d_loss_wtag = tf.reduce_mean(self.wtag_logits)
#         self.d_loss = (self.d_loss_fake + self.d_loss_wtag) * 0.5 - self.d_loss_real \
#                             + self.hparams.penalty_coef * gradient_penalty



    def _add_optim(self):
        tvars = tf.trainable_variables()
        self.d_vars = [var for var in tvars if 'discriminator' in var.name]
        self.g_vars = [var for var in tvars if 'generator' in var.name]

        self.global_step = tf.Variable(0, trainable=False)
        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.hparams.beta1).\
                        minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.hparams.beta1).\
                        minimize(self.g_loss, var_list=self.g_vars)


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

        test_tags = image_helper.get_test_tags()
        for batch_id, batch_data in image_helper.iter_images(batch_size=self.batch_size, 
                                                             epoches=self.epoches):
            num_epoch, num_batch = batch_id
            batch_images, batch_tags, batch_wtags = batch_data
            if (num_epoch == 0) and (num_batch < self.hparams.d_pretrain):
                # pre-train discriminator
                _, current_step, d_loss, d_accuarcy = sess.run(
                    [self.d_optim, self.global_step, self.d_loss, self.d_accuarcy], 
                    feed_dict={
                        self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                        self.real_imgs: batch_images,
                        self.tags: self.batch_tags,
                        self.wrong_tags: batch_wtags})
                if current_step == self.hparams.d_pretrain:
                    tf.logging.info("==== pre-train ==== current_step:{}, d_loss:{}, d_accuarcy:{}"\
                                    .format(current_step, d_loss, d_accuarcy))
            else:
                # optimize discriminator
                _, current_step, d_loss, d_accuarcy = sess.run(
                    [self.d_optim, self.global_step, self.d_loss, self.d_accuarcy], 
                    feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                               self.real_imgs: batch_images,
                               self.tags: batch_tags,
                               self.wrong_tags: batch_wtags})
                # import IPython
                # IPython.embed()
                # optimize generator
                if current_step % self.hparams.d_schedule == 0:
                    _, g_loss = sess.run(
                        [self.g_optim, self.g_loss], 
                        feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                                   self.tags: batch_tags})

                # summary
                if current_step % self.hparams.log_interval == 0:
#                     import IPython
#                     IPython.embed()
                    d_summary_str, g_summary_str = sess.run(
                        [d_summary_op, g_summary_op], 
                        feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                                   self.real_imgs: batch_images,
                                   self.tags: batch_tags,
                                   self.wrong_tags: batch_wtags})
                    summary_writer.add_summary(d_summary_str, current_step)
                    summary_writer.add_summary(g_summary_str, current_step)

                    tf.logging.info("step:{}, d_loss:{}, g_loss:{}, d_accuarcy:{}"\
                                    .format(current_step, d_loss, g_loss, d_accuarcy))

            if (num_epoch > 0) and (num_batch == 0):
                # generate images per epoch
                tf.logging.info("epoch:{} === generate images and save checkpoint".format(num_epoch))
                fake_imgs = sess.run(
                    self.fake_imgs, 
                    feed_dict={self.rand_noises: np.random.normal(size=[self.batch_size, self.noise_dim]),
                               self.tags: test_tags})
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

