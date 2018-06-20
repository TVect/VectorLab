import os
import time
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
    mode = tf.contrib.learn.ModeKeys.TRAIN
    use_attention = True
    input_reverse = True
    beam_width = 3  # 使用 beam_width 控制是否要进行 beam search
    
    SOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    
    vocab_size = 100
    emb_size = 100

    en_hidden_units = 100
    de_hidden_units = 100

    emb_matrix = None

    lr = 0.001
    decay_steps = 1000
    decay_rate = 0.96
    
    output_dir = os.path.abspath(os.path.join(os.path.curdir, "output"))


class BaseChatModel:
    
    def __init__(self, config):
        self.config = config
        self.build_graph()


    def build_graph(self):
        self.en_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="en_inputs")
        self.en_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="en_lengths")
        
        if self.config.mode != tf.contrib.learn.ModeKeys.INFER:
            self.de_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_inputs")
            self.de_outputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_outputs")
            self.de_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="de_lengths")
        
        
        # 是否要进行输入序列反转
        if self.config.input_reverse:
            self.final_en_inputs = tf.reverse_sequence(self.en_inputs, self.en_lengths, seq_dim=1)
        else:
            self.final_en_inputs = self.en_inputs

        # 是否使用预定义的 embedding_matrix
        if self.config.emb_matrix is not None:
            self.emb_matrix = tf.Variable(self.config.emb_matrix, trainable=False, dtype=tf.float32, name="emb_matrix")
        else:
            self.emb_matrix = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.emb_size], -1.0, 1.0), 
                dtype=tf.float32, name="emb_matrix")
        
        self.en_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.final_en_inputs, name="reverse_en_emb_inputs")
        if self.config.mode != tf.contrib.learn.ModeKeys.INFER:
            self.de_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_inputs, name="de_emb_inputs")
            self.de_emb_outputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_outputs, name="de_emb_outputs")
    
        with tf.variable_scope("encoder"):
            en_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.en_hidden_units)
            en_outputs, en_state = tf.nn.dynamic_rnn(
                en_cell, inputs=self.en_emb_inputs, sequence_length=self.en_lengths, dtype=tf.float32)
        
        with tf.variable_scope("decoder"):
            de_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.de_hidden_units)
            
            de_initial_state = en_state
            # 是否要使用 attention 机制
            if self.config.use_attention:
                if self.config.mode == tf.contrib.learn.ModeKeys.INFER:
                    '''
                    BeamSearchDecoder 和 AttentionWrapper 同时使用时, 需要确保:
                    1. encoder output 需要通过   @{tf.contrib.seq2seq.tile_batch} (NOT `tf.tile`) 复制 beam_width 次
                    2. The `batch_size` argument passed to the `zero_state` method of this
                        wrapper is equal to `true_batch_size * beam_width`.
                    3. The initial state created with `zero_state` above contains a
                        `cell_state` value containing properly tiled final state from the encoder.
                    '''
                    tiled_en_outputs = tf.contrib.seq2seq.tile_batch(en_outputs, multiplier=self.config.beam_width)
                    tiled_en_state = tf.contrib.seq2seq.tile_batch(en_state, multiplier=self.config.beam_width)
                    tiled_en_lengths = tf.contrib.seq2seq.tile_batch(self.en_lengths, multiplier=self.config.beam_width)
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.de_hidden_units,
                                                                            memory=tiled_en_outputs,
                                                                            memory_sequence_length=tiled_en_lengths)
                    de_cell = tf.contrib.seq2seq.AttentionWrapper(de_cell, attention_mechanism)
                    
                    de_initial_state = de_cell.zero_state(batch_size=tf.shape(self.en_inputs)[0] * self.config.beam_width, dtype=tf.float32)
                    de_initial_state = de_initial_state.clone(cell_state=tiled_en_state)
                else:
                    attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.de_hidden_units,
                                                                            memory=en_outputs,
                                                                            memory_sequence_length=self.en_lengths)
                    de_cell = tf.contrib.seq2seq.AttentionWrapper(de_cell, attention_mechanism)

                    de_initial_state = de_cell.zero_state(batch_size=tf.shape(self.en_inputs)[0], dtype=tf.float32)
                    de_initial_state = de_initial_state.clone(cell_state=en_state)

            # 判断模式是 TRAIN, EVAL 还是 INFER
            maximum_iterations = None
            projection_layer = tf.layers.Dense(self.config.vocab_size)

            if self.config.mode != tf.contrib.learn.ModeKeys.INFER:
                helper = tf.contrib.seq2seq.TrainingHelper(self.de_emb_inputs, self.de_lengths)
                decoder = tf.contrib.seq2seq.BasicDecoder(de_cell, helper, 
                                                          initial_state=de_initial_state, 
                                                          output_layer=projection_layer)
            else:
                start_tokens = tf.fill([tf.shape(self.en_inputs)[0]], self.config.SOS_TOKEN_ID)
                end_token = self.config.EOS_TOKEN_ID                
                # INFER时 限定解码的最大长度
                max_encoder_length = tf.reduce_max(self.en_lengths)
                maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * 2.0))

                # 是否进行 Beam Search 解码
                if self.config.beam_width <= 1:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.emb_matrix, start_tokens, end_token)
                    decoder = tf.contrib.seq2seq.BasicDecoder(de_cell, helper, 
                                                              initial_state=de_initial_state, 
                                                              output_layer=projection_layer)
                else:                    
                    # beam search
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=de_cell,
                                                                   embedding=self.emb_matrix,
                                                                   start_tokens=start_tokens,
                                                                   end_token=end_token,
                                                                   initial_state=de_initial_state,
                                                                   beam_width=self.config.beam_width,
                                                                   output_layer=projection_layer,
                                                                   length_penalty_weight=0.0)

            self.final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)

        if hasattr(self.final_outputs, "sample_id"):
            self.predictions = self.final_outputs.sample_id
        elif hasattr(self.final_outputs, "predicted_ids"):
            self.predictions = self.final_outputs.predicted_ids


    def _add_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.de_outputs, 
                                                              logits=self.final_outputs.rnn_output)
        target_weights = tf.sequence_mask(self.de_lengths, tf.shape(self.de_outputs)[1], dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.reduce_sum(loss * target_weights, axis=1), name="loss")


    def _add_train_op(self):
        self.global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.config.lr,  
                                                   self.global_steps,  
                                                   decay_steps=self.config.decay_steps,  
                                                   decay_rate=self.config.decay_rate,
                                                   staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_steps)


    def _add_saver(self):
        # checkpoint 相关
        self.checkpoint_dir = os.path.abspath(os.path.join(self.config.output_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


    def _add_summary(self):
        tf.summary.scalar("loss", self.loss)
        self.train_summary_op = tf.summary.merge_all()


    def train(self, sess, data_helper):
        assert self.config.mode == tf.contrib.learn.ModeKeys.TRAIN
        
        self._add_loss()
        self._add_train_op()
        self._add_summary()
        self._add_saver()
        train_summary_dir = os.path.join(self.config.output_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        for batch_en_inputs, batch_de_inputs, batch_de_outputs, batch_en_lengths, batch_de_lengths in data_helper.batch_iter():
            _, steps, loss, summaries = sess.run([self.train_op, self.global_steps, self.loss, self.train_summary_op], 
                                                 feed_dict={self.en_inputs: batch_en_inputs, 
                                                            self.de_inputs: batch_de_inputs,
                                                            self.de_outputs: batch_de_outputs[:, :max(batch_de_lengths)], 
                                                            self.en_lengths: batch_en_lengths, 
                                                            self.de_lengths: batch_de_lengths})
            train_summary_writer.add_summary(summaries, steps)
            print("training steps: %s, loss: %s" % (steps, loss))
            
            if steps % 100 == 0:
                self.saver.save(sess, self.checkpoint_prefix, global_step=steps)



    def eval(self, sess, en_inputs, de_inputs, de_outputs, en_lengths, de_lengths):
        assert self.config.mode == tf.contrib.learn.ModeKeys.EVAL
        loss = sess.run([self.loss], feed_dict={self.en_inputs: en_inputs,
                                                self.en_lengths: en_lengths,
                                                self.de_inputs: de_inputs,
                                                self.de_outputs: de_outputs,
                                                self.de_lengths: de_lengths})
        print("eval loss: ", loss)
        
    
    def infer(self, sess, en_inputs, en_lengths):
        assert self.config.mode == tf.contrib.learn.ModeKeys.INFER
        self._add_saver()
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        preds = sess.run(self.predictions, feed_dict={self.en_inputs: en_inputs,
                                                        self.en_lengths: en_lengths})
        return preds

