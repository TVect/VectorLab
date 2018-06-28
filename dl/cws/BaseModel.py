import tensorflow as tf

class BaseModel:
    
    def __init__(self, hparams):
        self.hparams = hparams
        self.build()


    def build(self):
        self._add_placeholder()
        self._add_embedding()
        self._add_predictions()
    
    
    def _add_placeholder(self):
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input")
        self.length_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="length")
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name="label")
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout")
        self.rnn_dropout_placeholder = tf.placeholder(dtype=tf.float32, name="rnn_dropout")
        

    def _add_embedding(self):
        self.embedding_matrix = tf.get_variable(name="embedding_matrix", 
                                                shape=[self.hparams.vocab_size, self.hparams.embed_dim], 
                                                dtype=tf.float32, 
                                                initializer=tf.contrib.layers.xavier_initializer())
        self.input_embedding = tf.nn.embedding_lookup(params=self.embedding_matrix, 
                                                      ids=self.input_placeholder, 
                                                      name="embeddings")


    def _add_predictions(self):
        def gen_lstm_cell():
            return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=self.hparams.num_units), 
                                                 output_keep_prob=self.rnn_dropout_placeholder)

        # 多层 lstm
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [gen_lstm_cell() for _ in range(self.hparams.num_lstm_layers)], 
            [gen_lstm_cell() for _ in range(self.hparams.num_lstm_layers)], 
            inputs=self.input_embedding, 
            sequence_length=self.length_placeholder,
            dtype=tf.float32)
        
        # project layer
        with tf.name_scope("proj_layer"):
            W_proj = tf.get_variable(name="W_proj", 
                                     shape=[outputs.shape[-1], self.hparams.tags_size], 
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_proj = tf.get_variable(name="b_proj",
                                     shape=[self.hparams.tags_size],
                                     initializer=tf.zeros_initializer())
            
            unary_scores = tf.nn.relu(tf.nn.dropout(tf.nn.xw_plus_b(tf.reshape(outputs, [-1, outputs.shape[-1]]), W_proj, b_proj), 
                                                   keep_prob=self.dropout_placeholder))
        unary_scores = tf.reshape(unary_scores, [-1, tf.shape(outputs)[1], self.hparams.tags_size], 
                                  name="unary_scores")

        # crf layer
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores, 
                                                                              self.label_placeholder, 
                                                                              self.length_placeholder)


    def train(self, sess, ):
        pass
    
    
    def eval(self):
        pass
    
    
    def infer(self):
        pass

