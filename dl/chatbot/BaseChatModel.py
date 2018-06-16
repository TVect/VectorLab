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
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Config:
    # Standard names for model modes: (TRAIN, 'train'), (EVAL, 'eval'), (INFER, 'infer')
    mode = tf.contrib.learn.ModeKeys.INFER
    
    SOS_TOKEN_ID = 0
    EOS_TOKEN_ID = 1
    
    vocab_size = 100
    emb_size = 100

    en_hidden_units = 100
    de_hidden_units = 100

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
        self.en_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="en_lengths")
        self.de_outputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_outputs")
        self.de_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="de_lengths")

        if self.config.emb_matrix is not None:
            self.emb_matrix = tf.Variable(self.config.emb_matrix, trainable=False, name="emb_matrix")
        else:
            self.emb_matrix = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.emb_size], -1.0, 1.0), 
                dtype=tf.float32, name="emb_matrix")
        self.en_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.en_inputs, name="en_emb_inputs")
        self.de_emb_outputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_outputs, name="de_emb_outputs")
    
        with tf.variable_scope("encoder"):
            en_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.en_hidden_units)
            en_outputs, en_state = tf.nn.dynamic_rnn(
                en_cell, inputs=self.en_emb_inputs, sequence_length=self.en_lengths, dtype=tf.float32)
        
        with tf.variable_scope("decoder"):
            de_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.de_hidden_units)
            if self.config.mode == tf.contrib.learn.ModeKeys.TRAIN:
                helper = tf.contrib.seq2seq.TrainingHelper(self.de_emb_outputs, self.de_lengths)
            elif self.config.mode == tf.contrib.learn.ModeKeys.EVAL:
                pass
            elif self.config.mode == tf.contrib.learn.ModeKeys.INFER:
                start_tokens = tf.fill([tf.shape(self.en_inputs)[0]], self.config.SOS_TOKEN_ID)
                end_token = self.config.EOS_TOKEN_ID
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.emb_matrix, start_tokens, end_token)

            helper = tf.contrib.seq2seq.TrainingHelper(self.de_emb_outputs, self.de_lengths)
            projection_layer = tf.layers.Dense(self.config.vocab_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(de_cell, helper, 
                                                      initial_state=en_state, 
                                                      output_layer=projection_layer)
            final_outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder)
            
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.de_outputs, 
                                                              logits=final_outputs.rnn_output)
        target_weights = tf.sequence_mask(self.de_lengths, tf.shape(self.de_outputs)[1], dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(loss * target_weights, axis=1), name="loss")


    @lazy_property
    def loss(self):
        pass


    @lazy_property
    def optimizer(self):
        # train_op = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)
        pass


    def fit(self, sess):
        sess.run(tf.global_variables_initializer())
