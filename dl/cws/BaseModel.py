import os
import tensorflow as tf
from input_pipeline import InputPipeline

class BaseModel:
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.build()


    def build(self):
        # self._add_placeholder()
        self._add_input_pipeline()
        self._add_embedding()
        self._add_predictions()
        self._add_saver()


    def _add_input_pipeline(self):
        self.input_pipeline = InputPipeline(is_user_input=self.hparams.is_user_input,
                                            record_file=self.hparams.record_file)
        self.inputs, self.in_lengths, self.labels = self.input_pipeline.get_inputs()

        self.dropout_ratio = tf.Variable(self.hparams.keep_ratio, trainable=False,
                                         name="dropout")
        self.rnn_dropout_ratio = tf.Variable(self.hparams.rnn_output_keep_prob, trainable=False,
                                             name="rnn_dropout")


    def _add_embedding(self):
        self.embedding_matrix = tf.get_variable(name="embedding_matrix", 
                                                shape=[self.hparams.vocab_size, self.hparams.embed_dim], 
                                                dtype=tf.float32, 
                                                initializer=tf.contrib.layers.xavier_initializer())
        self.input_embedding = tf.nn.embedding_lookup(params=self.embedding_matrix, 
                                                      ids=self.inputs, 
                                                      name="embeddings")


    def _add_predictions(self):
        def gen_lstm_cell():
            return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=self.hparams.num_units), 
                                                 output_keep_prob=self.rnn_dropout_ratio)

        # 多层 lstm
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [gen_lstm_cell() for _ in range(self.hparams.num_lstm_layers)], 
            [gen_lstm_cell() for _ in range(self.hparams.num_lstm_layers)], 
            inputs=self.input_embedding, 
            sequence_length=self.in_lengths,
            dtype=tf.float32)
        
        # project layer
        with tf.name_scope("proj_layer"):
            W_proj = tf.get_variable(name="W_proj", 
                                     shape=[outputs.shape[-1], self.hparams.tags_size], 
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_proj = tf.get_variable(name="b_proj",
                                     shape=[self.hparams.tags_size],
                                     initializer=tf.zeros_initializer())
            
            unary_potentials = tf.nn.relu(tf.nn.dropout(tf.nn.xw_plus_b(tf.reshape(outputs, [-1, outputs.shape[-1]]), W_proj, b_proj), 
                                                        keep_prob=self.dropout_ratio))
            self.unary_potentials = tf.reshape(unary_potentials, [-1, tf.shape(outputs)[1], self.hparams.tags_size], 
                                               name="unary_potentials")

        # crf layer
        self.transition_params = tf.get_variable(name="transitions", 
                                                 shape=[self.hparams.tags_size, self.hparams.tags_size])
        if not self.hparams.is_user_input:
            self.log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.unary_potentials, 
                                                                       self.labels, 
                                                                       self.in_lengths,
                                                                       transition_params=self.transition_params)


    def _add_saver(self):
        # checkpoint 相关
        self.checkpoint_dir = os.path.abspath(os.path.join(self.hparams.output_dir, "checkpoints"))
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)


    def train(self, sess):
        assert self.hparams.is_user_input == False
        
        global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.hparams.lr,  
                                                   global_steps,  
                                                   decay_steps=self.hparams.decay_steps,  
                                                   decay_rate=self.hparams.decay_rate,
                                                   staircase=True)
        loss = tf.reduce_mean(-self.log_likelihood)
        train_op = tf.train.AdamOptimizer(learning_rate)\
                            .minimize(loss, global_step=global_steps)
        tf.summary.scalar("loss", loss)
        train_summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(self.hparams.output_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                _, _steps, _loss, _summaries = sess.run([train_op, global_steps, loss, train_summary_op])
                
                train_summary_writer.add_summary(_summaries, _steps)
                
                if _steps % 10 == 0:
                    print("training steps: %s, loss: %s" % (_steps, _loss))
                if _steps % 100 == 0:
                    self.saver.save(sess, self.checkpoint_prefix, global_step=_steps)
        except Exception as e:
            print(e)
            coord.request_stop()
        finally:
            coord.join(threads)
        

    def eval(self):
        pass
    
    
    def infer(self, sess, user_inputs, in_lengths):
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        
        feed_dict = self.input_pipeline.feed_user_dict(user_inputs, in_lengths)
        feed_dict[self.dropout_ratio] = 1.0
        feed_dict[self.rnn_dropout_ratio] = 1.0
        decode_tags, best_score = tf.contrib.crf.crf_decode(self.unary_potentials, 
                                                            self.transition_params, 
                                                            self.in_lengths)
        # sess.run(tf.global_variables_initializer())
        tags, scores = sess.run([decode_tags, best_score], feed_dict=feed_dict)
        return tags, scores
        