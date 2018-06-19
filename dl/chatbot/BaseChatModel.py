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
    
    input_reverse = True
    
    SOS_TOKEN_ID = 0
    EOS_TOKEN_ID = 1
    
    vocab_size = 100
    emb_size = 100

    en_hidden_units = 100
    de_hidden_units = 100

    emb_matrix = None

    lr = 0.001
    decay_steps = 1000
    decay_rate = 0.96


class BaseChatModel:
    
    def __init__(self, config):
        self.config = config
        self.build_graph()


    def build_graph(self):
        self.en_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="en_inputs")
        self.en_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="en_lengths")
        self.de_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_inputs")
        self.de_outputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="de_outputs")
        self.de_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="de_lengths")

        if self.config.input_reverse:
            self.final_en_inputs = tf.reverse_sequence(self.en_inputs, self.en_lengths, 
                                                       seq_dim=1)
        else:
            self.final_en_inputs = self.en_inputs

        if self.config.emb_matrix is not None:
            self.emb_matrix = tf.Variable(self.config.emb_matrix, trainable=False, dtype=tf.float32, name="emb_matrix")
        else:
            self.emb_matrix = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.emb_size], -1.0, 1.0), 
                dtype=tf.float32, name="emb_matrix")
        self.en_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.final_en_inputs, name="reverse_en_emb_inputs")
        self.de_emb_inputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_inputs, name="de_emb_inputs")
        self.de_emb_outputs = tf.nn.embedding_lookup(self.emb_matrix, self.de_outputs, name="de_emb_outputs")
    
        with tf.variable_scope("encoder"):
            en_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.en_hidden_units)
            en_outputs, en_state = tf.nn.dynamic_rnn(
                en_cell, inputs=self.en_emb_inputs, sequence_length=self.en_lengths, dtype=tf.float32)
        
        with tf.variable_scope("decoder"):
            de_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.de_hidden_units)
            maximum_iterations = None
            if self.config.mode == tf.contrib.learn.ModeKeys.TRAIN:
                helper = tf.contrib.seq2seq.TrainingHelper(self.de_emb_inputs, self.de_lengths)
            elif self.config.mode == tf.contrib.learn.ModeKeys.EVAL:
                helper = tf.contrib.seq2seq.TrainingHelper(self.de_emb_inputs, self.de_lengths)
            elif self.config.mode == tf.contrib.learn.ModeKeys.INFER:
                start_tokens = tf.fill([tf.shape(self.de_outputs)[0]], self.config.SOS_TOKEN_ID)
                end_token = self.config.EOS_TOKEN_ID
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.emb_matrix, start_tokens, end_token)
                # INFER时 限定解码的最大长度
                max_encoder_length = tf.reduce_max(self.en_lengths)
                maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * 2.0))

            projection_layer = tf.layers.Dense(self.config.vocab_size)
            decoder = tf.contrib.seq2seq.BasicDecoder(de_cell, helper, 
                                                      initial_state=en_state, 
                                                      output_layer=projection_layer)
            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations)
        
        self.logits = final_outputs.rnn_output
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.de_outputs, 
                                                              logits=final_outputs.rnn_output)
        target_weights = tf.sequence_mask(self.de_lengths, tf.shape(self.de_outputs)[1], dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.reduce_sum(loss * target_weights, axis=1), name="loss")

        self.global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.config.lr,  
                                                   self.global_steps,  
                                                   decay_steps=self.config.decay_steps,  
                                                   decay_rate=self.config.decay_rate,
                                                   staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_steps)


    def fit(self, sess, data_helper):
        for batch_en_inputs, batch_de_inputs, batch_de_outputs, batch_en_lengths, batch_de_lengths in data_helper.batch_iter():
            _, steps, loss = sess.run([self.train_op, self.global_steps, self.loss], 
                               feed_dict={self.en_inputs: batch_en_inputs, 
                                          self.de_inputs: batch_de_inputs,
                                          self.de_outputs: batch_de_outputs[:, :max(batch_de_lengths)], 
                                          self.en_lengths: batch_en_lengths, 
                                          self.de_lengths: batch_de_lengths})

            print("steps: %s, loss: %s" % (steps, loss))



if __name__ == "__main__":
    from DataHelper import DataHelper
    from MyVocabProcessor import MyVocabProcessor
    vocab_processor = MyVocabProcessor(vcb_file="embedding/wordvecs.vcb", 
                                       vec_file="embedding/wordvecs.txt",
                                       max_document_length=40)
    data_helper = DataHelper(vocab_processor)
    
    npy_names = {"en_inputs": "data/parsed_data/en_inputs.npy",
                 "de_inputs": "data/parsed_data/de_inputs.npy", 
                 "de_outputs": "data/parsed_data/de_outputs.npy", 
                 "en_lengths": "data/parsed_data/en_lengths.npy", 
                 "de_lengths": "data/parsed_data/de_lengths.npy"}
    data_helper.gen_from_npy(npy_names)
    
    with tf.Graph().as_default():
        config = Config()
        config.vocab_size = len(data_helper.vocab_processor.vocab_table)
        config.emb_size = data_helper.vocab_processor.vector_size
        config.emb_matrix = data_helper.vocab_processor.get_embedding_matrix()
        config.SOS_TOKEN_ID = data_helper.vocab_processor.SOS_ID
        config.EOS_TOKEN_ID = data_helper.vocab_processor.EOS_ID

        chat_model = BaseChatModel(config)
        
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            chat_model.fit(session, data_helper)
            
        
