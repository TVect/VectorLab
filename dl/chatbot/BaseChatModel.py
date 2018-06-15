import tensorflow as tf

'''
基本的 Chatbot 模型
'''

import functools

def lazy_property(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Config:
    # Standard names for model modes: (TRAIN, 'train'), (EVAL, 'eval'), (INFER, 'infer')
    mode = tf.contrib.learn.ModeKeys.INFER
    
    vocab_size = None
    emb_size = None

    en_hidden_units = None
    de_hidden_units = None

    emb_matrix = None

    learning_rate = 0.001



class BaseChatModel:
    
    def __init__(self, config):
        self.config = config

        self.prediction
        self.loss
        self.optimizer


    @lazy_property
    def prediction(self):
        self.en_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="en_inputs")
        self.en_lengths = tf.placeholder(type=tf.int32, shape=[None], name="en_lengths")
        self.de_outputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_outputs")
        
        if self.config.emb_matrix is not None:
            self.emb_matrix = tf.Variable(self.config.emb_matrix, trainable=False, name="emb_matrix")
        else:
            self.emb_matrix = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.input_emb_size], -1.0, 1.0), 
                dtype=tf.float32, name="emb_matrix")
        self.en_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.en_inputs, name="en_emb_inputs")
        self.de_emb_outputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_outputs, name="de_emb_outputs")
    
        with tf.variable_scope("encoder"):
            en_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.en_hidden_units)
            en_outputs, en_state = tf.nn.dynamic_rnn(
                en_cell, inputs=self.en_emb_inputs, seq_length=self.en_lengths)
        
        with tf.variable_scope("decoder"):
            de_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.de_hidden_units)
            helper = tf.contrib.seq2seq.TrainingHelper()
            tf.contrib.seq2seq.BasicDecoder(de_cell, helper, initial_state=en_state.h)


    @lazy_property
    def loss(self):
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=decoder_logits)

        loss = tf.reduce_mean(stepwise_cross_entropy)
        return loss


    @lazy_property
    def optimize(self):
        train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)


    def fit(self, sess):
        sess.run(tf.global_variables_initializer())


        