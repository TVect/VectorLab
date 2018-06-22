import os
import time
import tensorflow as tf

'''
基本的 Chatbot 模型
'''


class Config:
    # Standard names for model modes: (TRAIN, 'train'), (EVAL, 'eval'), (INFER, 'infer')
    mode = tf.contrib.learn.ModeKeys.TRAIN
    use_attention = True
    input_reverse = False
    beam_width = 10  # 使用 beam_width 控制是否要进行 beam search
    
    SOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    
    vocab_size = 100
    emb_size = 300

    en_hidden_units = 300
    de_hidden_units = 300

    emb_matrix = None

    lr = 0.001
    decay_steps = 1000
    decay_rate = 0.96
    
    output_dir = os.path.abspath(os.path.join(os.path.curdir, "output_char"))


class BaseChatModel:
    
    def __init__(self, config):
        self.config = config
        self.build_graph()
    
    
    def build_graph(self):
        self._add_placeholder()
        self._build_encoder()
        self._build_decoder()
        self._add_saver()
        
    
    def _add_placeholder(self):
        self.en_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="en_inputs")
        self.en_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="en_lengths")
        
        if self.config.mode != tf.contrib.learn.ModeKeys.INFER:
            self.de_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_inputs")
            self.de_outputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_outputs")
            self.de_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="de_lengths")
        

    def _build_encoder(self):
        self.emb_matrix = tf.Variable(
            tf.random_uniform([self.config.vocab_size, self.config.emb_size], -1.0, 1.0), 
            dtype=tf.float32, name="emb_matrix")
        
        self.en_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.en_inputs, name="en_emb_inputs")
        if self.config.mode != tf.contrib.learn.ModeKeys.INFER:
            self.de_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_inputs, name="de_emb_inputs")
            self.de_emb_outputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_outputs, name="de_emb_outputs")
    
        with tf.variable_scope("encoder"):
            en_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.en_hidden_units)
            self.en_outputs, self.en_state = tf.nn.dynamic_rnn(en_cell, 
                                                               inputs=self.en_emb_inputs, 
                                                               sequence_length=self.en_lengths, 
                                                               dtype=tf.float32)
            

    def _build_decoder(self):
        maximum_iterations = None
        projection_layer = tf.layers.Dense(self.config.vocab_size)
        de_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.de_hidden_units)
        
        if self.config.mode == tf.contrib.learn.ModeKeys.TRAIN:
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.de_hidden_units,
                                                                    memory=self.en_outputs,
                                                                    memory_sequence_length=self.en_lengths)
            de_cell = tf.contrib.seq2seq.AttentionWrapper(de_cell, attention_mechanism)

            de_initial_state = de_cell.zero_state(batch_size=tf.shape(self.en_inputs)[0], dtype=tf.float32)
            de_initial_state = de_initial_state.clone(cell_state=self.en_state)            
    
            train_helper = tf.contrib.seq2seq.TrainingHelper(self.de_emb_inputs, self.de_lengths)
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=de_cell, 
                                                      helper=train_helper, 
                                                      initial_state=de_initial_state, 
                                                      output_layer=projection_layer)
            self.final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                            maximum_iterations=maximum_iterations)
            self.logits = self.final_outputs.rnn_output
            target_weights = tf.sequence_mask(self.de_lengths, 
                                              tf.shape(self.de_outputs)[1], 
                                              dtype=tf.float32)
            self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits,
                                                         targets=self.de_outputs,
                                                         weights=target_weights)
        elif self.config.mode == tf.contrib.learn.ModeKeys.INFER:
            start_tokens = tf.fill([tf.shape(self.en_inputs)[0]], self.config.SOS_TOKEN_ID)
            end_token = self.config.EOS_TOKEN_ID                
            # INFER时 限定解码的最大长度
            max_encoder_length = tf.reduce_max(self.en_lengths)
            maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * 2.0))         
            # beam search
            '''
            initial_state = tf.contrib.seq2seq.tile_batch(self.en_state, multiplier=self.config.beam_width)
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=de_cell,
                                                           embedding=self.emb_matrix,
                                                           start_tokens=start_tokens,
                                                           end_token=end_token,
                                                           initial_state=initial_state,
                                                           beam_width=self.config.beam_width,
                                                           output_layer=projection_layer)
            '''
            tiled_en_outputs = tf.contrib.seq2seq.tile_batch(self.en_outputs, multiplier=self.config.beam_width)
            tiled_en_state = tf.contrib.seq2seq.tile_batch(self.en_state, multiplier=self.config.beam_width)
            tiled_en_lengths = tf.contrib.seq2seq.tile_batch(self.en_lengths, multiplier=self.config.beam_width)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.config.de_hidden_units,
                                                                    memory=tiled_en_outputs,
                                                                    memory_sequence_length=tiled_en_lengths)
            de_cell = tf.contrib.seq2seq.AttentionWrapper(de_cell, attention_mechanism)
            
            de_initial_state = de_cell.zero_state(batch_size=tf.shape(self.en_inputs)[0] * self.config.beam_width, dtype=tf.float32)
            de_initial_state = de_initial_state.clone(cell_state=tiled_en_state)
            
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=de_cell,
                                                           embedding=self.emb_matrix,
                                                           start_tokens=start_tokens,
                                                           end_token=end_token,
                                                           initial_state=de_initial_state,
                                                           beam_width=self.config.beam_width,
                                                           output_layer=projection_layer)
            self.final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, 
                                                            maximum_iterations=maximum_iterations)
            self.predictions = self.final_outputs.predicted_ids


    def _add_saver(self):
        # checkpoint 相关
        self.checkpoint_dir = os.path.abspath(os.path.join(self.config.output_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


    def train(self, sess, data_helper):
        assert self.config.mode == tf.contrib.learn.ModeKeys.TRAIN
        
        self.global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.config.lr,  
                                                   self.global_steps,  
                                                   decay_steps=self.config.decay_steps,  
                                                   decay_rate=self.config.decay_rate,
                                                   staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, 
                                                                       global_step=self.global_steps)
        tf.summary.scalar("loss", self.loss)
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(self.config.output_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        init = tf.global_variables_initializer()
        sess.run(init)

        for batch_en_inputs, batch_de_inputs, batch_de_outputs, batch_en_lengths, batch_de_lengths in data_helper.batch_iter():
            _, steps, loss, summaries = sess.run([self.train_op, self.global_steps, self.loss, train_summary_op], 
                                                 feed_dict={self.en_inputs: batch_en_inputs, 
                                                            self.de_inputs: batch_de_inputs,
                                                            self.de_outputs: batch_de_outputs[:, :max(batch_de_lengths)], 
                                                            self.en_lengths: batch_en_lengths, 
                                                            self.de_lengths: batch_de_lengths})
            train_summary_writer.add_summary(summaries, steps)
            print("training steps: %s, loss: %s" % (steps, loss))
            
            if steps % 100 == 0:
                self.saver.save(sess, self.checkpoint_prefix, global_step=steps)
        
    
    def infer(self, sess, en_inputs, en_lengths):
        assert self.config.mode == tf.contrib.learn.ModeKeys.INFER
        
        init = tf.global_variables_initializer()
        sess.run(init)
        
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)

        preds = sess.run(self.predictions, feed_dict={self.en_inputs: en_inputs,
                                                        self.en_lengths: en_lengths})
        return preds

