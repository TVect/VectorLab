import os
import tensorflow as tf
from input_pipeline import InputPipeline
from layer_utils import idcnn_layer, lstm_layer


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
                                            # record_file=self.hparams.record_file,
                                            train_record=self.hparams.train_record,
                                            valid_record=self.hparams.valid_record, 
                                            batch_size=self.hparams.batch_size,
                                            num_epochs=self.hparams.num_epochs)
        self.in_lengths, self.inputs, self.labels = self.input_pipeline.get_inputs()

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
        if self.hparams.layer_mode == "lstm":
            print("============== use lstm ===============")
            outputs = lstm_layer(self.input_embedding, 
                                 self.in_lengths, 
                                 lstm_layers=self.hparams.lstm_layers, 
                                 lstm_units=self.hparams.lstm_units,
                                 rnn_dropout_ratio=self.rnn_dropout_ratio)
        elif self.hparams.layer_mode == "idcnn":
            print("============== use idcnn ===============")
            outputs = idcnn_layer(self.input_embedding, 
                                  block_cnt=self.hparams.idcnn_blocks, 
                                  layer_per_block=self.hparams.idcnn_layerpb)
        
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
        summary_op = tf.summary.merge_all()
        train_summary_dir = os.path.join(self.hparams.output_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        
        valid_summary_dir = os.path.join(self.hparams.output_dir, "summaries", "valid")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)


        test_var = tf.Variable(True)        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sess.run(self.input_pipeline.train_init)
        sess.run(self.input_pipeline.valid_init)

        try:
            while True:
                _, _steps, _loss, _summaries = sess.run([train_op, global_steps, loss, summary_op],
                                                        feed_dict={self.input_pipeline.train_or_valid: "train"})
                
                if _steps % self.hparams.traininfo_every == 0:
                    train_summary_writer.add_summary(_summaries, _steps)
                    print("training steps: %s, loss: %s" % (_steps, _loss))

                if _steps % self.hparams.evaluate_every == 0:
                    # self.input_pipeline.active_mode("valid")
                    valid_loss, valid_summary = sess.run([loss, summary_op], 
                                                         feed_dict={self.dropout_ratio: 1.0,
                                                                    self.rnn_dropout_ratio: 1.0,
                                                                    self.input_pipeline.train_or_valid: "valid"})
                    valid_summary_writer.add_summary(valid_summary, _steps)
                    print("====== valid steps: %s, loss: %s ======" % (_steps, valid_loss))
                    # self.input_pipeline.active_mode("train")
                
                if _steps % self.hparams.checkpoint_every == 0:
                    self.saver.save(sess, self.checkpoint_prefix, global_step=_steps)
        except tf.errors.OutOfRangeError as e:
            print("Train: Out of Range")
        except Exception as e:
            print("============= Train: Got An Exception =============")
            print(e)


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
        