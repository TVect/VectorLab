import numpy as np
import tensorflow as tf


class Config:
    vocab_size = 10000
    embed_dim = 100
    num_units = 50
    # dense_units = 50
    keep_ratio = 0.5
    rnn_output_keep_prob = 0.5
    
    # rnn_output_keep_prob = 0.5
    # rnn_input_keep_prob = 1.0
    # rnn_state_keep_prob = 1.0
    
    # lstm 层数
    num_lstm_layers = 3
    # 每一个全连接的 units
    dense_units_per_layer = [64, 32]    
    # l2 正则化
    l2_ratio = 1.0
    # 学习率相关
    lr = 0.001
    decay_steps = 1000
    decay_rate = 0.96
    
    epoch = 50
    batch_size = 128

    embedding_matrix = None

class TfSentiAnalyzer:
    
    def __init__(self, config):
        self.config = config
        self.build()
    
    def build(self):
        self.add_placeholders()
        self.add_embedding()
        self.add_predictions()
        self.add_loss_op()
        self.add_train_op()

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input")
        self.length_placeholder = tf.placeholder(dtype=tf.int32, shape=[None], name="length")
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None], name="label")
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name="dropout")
        self.rnn_dropout_placeholder = tf.placeholder(dtype=tf.float32, name="rnn_dropout")
    
    def add_embedding(self):
        if self.config.embedding_matrix is not None:
            print("using existed matrix")
            self.embedding_matrix = tf.Variable(self.config.embedding_matrix, 
                                                trainable=False,
                                                name="embedding_matrix")
        else:
            self.embedding_matrix = tf.get_variable(name="embedding_matrix", 
                                                    shape=[self.config.vocab_size, self.config.embed_dim], 
                                                    dtype=tf.float32, 
                                                    initializer=tf.contrib.layers.xavier_initializer())
        self.input_embedding = tf.nn.embedding_lookup(params=self.embedding_matrix, 
                                                      ids=self.input_placeholder, 
                                                      name="embeddings")
        
    def add_predictions(self):
        def gen_lstm_cell():
            return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_units), 
                                                 output_keep_prob=self.rnn_dropout_placeholder)
        ''' 
        cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_units)
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=self.rnn_dropout_placeholder)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_units)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.rnn_dropout_placeholder)
        '''
        # 多层 lstm
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([gen_lstm_cell() for _ in range(self.config.num_lstm_layers)], 
                                                                                                   [gen_lstm_cell() for _ in range(self.config.num_lstm_layers)], 
                                                                                                   inputs=self.input_embedding, 
                                                                                                   sequence_length=self.length_placeholder,
                                                                                                   dtype=tf.float32)
        out = output_state_fw[-1].h

        self.dense_weights = []

        for dense_id, dense_units in enumerate(self.config.dense_units_per_layer):
            print("dense_id:", dense_id, "dense_units:", dense_units)
            with tf.name_scope("dense_{}".format(dense_id)):
                W_dense = tf.get_variable(name="W_dense_{}".format(dense_id), 
                                          shape=[out.shape[1], dense_units], 
                                          initializer=tf.contrib.layers.xavier_initializer())
                b_dense = tf.get_variable(name="b_dense_{}".format(dense_id),
                                          shape=[dense_units],
                                          initializer=tf.zeros_initializer())
                out = tf.nn.relu(tf.nn.dropout(tf.matmul(out, W_dense) + b_dense, 
                                               keep_prob=self.dropout_placeholder), 
                                 name="out_{}".format(dense_id))
                self.dense_weights.append(W_dense)

        '''
        with tf.name_scope("dense_1"):
            W_dense = tf.get_variable(name="W_dense_1", 
                                      shape=[out.shape[1], self.config.dense_units_per_layer[0]], 
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_dense = tf.get_variable(name="b_dense_1",
                                      shape=[self.config.dense_units_per_layer[0]],
                                      initializer=tf.zeros_initializer())
            out = tf.nn.relu(tf.nn.dropout(tf.matmul(out, W_dense) + b_dense, 
                                           keep_prob=self.dropout_placeholder), 
                             name="out_1")        
        '''
        with tf.name_scope("output"):
            W_output = tf.get_variable(name="W_output", 
                                       shape=[out.shape[1], 1], 
                                       initializer=tf.contrib.layers.xavier_initializer())
            b_output = tf.get_variable(name="b_output",
                                       shape=[1],
                                       initializer=tf.zeros_initializer())
            output = tf.matmul(out, W_output) + b_output

        self.output = tf.reshape(output, [-1])
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.output>0, tf.int32), 
                                                        tf.cast(self.label_placeholder, tf.int32)), 
                                               tf.float32), 
                                       name="accuracy")
    
    def add_loss_op(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label_placeholder, 
                                                       logits=self.output)
        self.loss = tf.reduce_mean(loss)

        for weights in self.dense_weights:
            self.loss += self.config.l2_ratio * tf.nn.l2_loss(weights)


    def add_train_op(self):
        self.global_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.config.lr,  
                                                   self.global_steps,  
                                                   decay_steps=self.config.decay_steps,  
                                                   decay_rate=self.config.decay_rate,
                                                   staircase=True)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_steps)


    def fit(self, session, train, dev, saver=None):
        X_input, y_input = train
        input_size = len(y_input)
        print("------------ input_size: %s ------------" % input_size)
        if dev:
            X_dev, y_dev = dev
            dev_size = len(y_dev)
            X_dev_length = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in X_dev])
            print("------------ dev_size: %s ------------" % dev_size)

        sentiment_length = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in X_input])
        num_batch = int((input_size-1)/self.config.batch_size)+1
        for loop in range(self.config.epoch):
            random_ids = np.random.permutation(input_size)
            X_shuffled = X_input[random_ids]
            y_shuffled = y_input[random_ids]
            length_shuffled = sentiment_length[random_ids]

            for batch_id in range(num_batch):
                start_id = batch_id*self.config.batch_size
                end_id = min((batch_id+1)*self.config.batch_size, input_size)
                _, global_step, loss, acc = session.run([self.train_op, self.global_steps, self.loss, self.accuracy], 
                            feed_dict={self.input_placeholder: X_shuffled[start_id : end_id],
                                       self.label_placeholder: y_shuffled[start_id : end_id],
                                       self.length_placeholder: length_shuffled[start_id : end_id],
                                       self.dropout_placeholder: self.config.keep_ratio,
                                       self.rnn_dropout_placeholder: self.config.rnn_output_keep_prob})
                if global_step % 100 == 0:
                    print("step:", global_step, "loss:", loss, "acc:", acc)
                if (global_step % 1000 == 0) and (dev is not None): 
                    total_true = 0
                    total_cnt = 0
                    dev_batch_size = 100
                    for dev_batch_id in range(int((dev_size-1)/dev_batch_size)+1):
                        start_id = dev_batch_id * dev_batch_size
                        end_id = min((dev_batch_id+1) * dev_batch_size, dev_size)
                        loss, acc = session.run([self.loss, self.accuracy], 
                                                feed_dict={self.input_placeholder: X_dev[start_id : end_id],
                                                           self.label_placeholder: y_dev[start_id : end_id],
                                                           self.length_placeholder: X_dev_length[start_id : end_id],
                                                           self.dropout_placeholder: 1.0,
                                                           self.rnn_dropout_placeholder: 1.0})
                        # print("dev_batch_id:", dev_batch_id, "start_id:", start_id, "end_id:", end_id, "acc:", acc)
                        total_cnt += (end_id - start_id)
                        total_true += (end_id - start_id) * acc
                    average_acc = total_true / total_cnt
                    print("--------------- total_true: %s, total_cnt: %s, acc: %s ---------------" % (total_true, total_cnt, average_acc))


if __name__ == "__main__":
    from DataHelper import DataHelper
    helper = DataHelper()
    helper.build()

    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format("word2vec.stem.wv")
    
    embedding_matrix = [model.word_vec(wd) if wd in model.wv else [0]*model.vector_size 
                        for wd in helper.vocab_processor.vocabulary_._reverse_mapping]
    
    X_train, X_test, y_train, y_test = helper.get_train_test_split()

    with tf.Graph().as_default():
        config = Config()
        config.vocab_size = len(helper.vocab_processor.vocabulary_)
        config.embedding_matrix = embedding_matrix

        analyzer = TfSentiAnalyzer(config)
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            analyzer.fit(session, (X_train, y_train), (X_test, y_test), None)
