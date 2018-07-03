import tensorflow as tf

class InputPipeline:
    
    def __init__(self, 
                 is_user_input=False, 
                 train_record=None,
                 valid_record=None, 
                 batch_size=128, 
                 num_epochs=10):
        '''
        @param is_user_input: 接收的是否是用户输入
        @param train_record: 用于训练的tf_record文件
        @param valid_record: 用于验证的tf_record文件
        @param batch_size: 每个batch的大小
        @param num_epochs: 控制训练几个epoch
        '''
        self.is_user_input = is_user_input
        if not self.is_user_input:
            assert train_record is not None
            assert valid_record is not None
            self.train_init, self.train_iter = self.build_pipeline(
                train_record, batch_size, num_epochs, True, scope_name="train_data")
            self.valid_init, self.valid_iter = self.build_pipeline(
                valid_record, batch_size*10, num_epochs, False, scope_name="valid_data")
            
            self.train_or_valid = tf.Variable("train", dtype=tf.string)
            # self.train_or_valid = tf.Variable(True, dtype=tf.bool)


    def get_inputs(self):
        with tf.variable_scope("input_scope"):
            if self.is_user_input:
                return self.get_user_inputs()
            else:
                record_inputs = self.get_record_inputs()
                return tf.cast(record_inputs[0], tf.int32, name="in_lengths"), \
                    tf.cast(record_inputs[1], tf.int32, name="inputs"), \
                    tf.cast(record_inputs[2], tf.int32, name="labels")


    def get_user_inputs(self):
        self.user_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="user_inputs")
        self.in_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="in_lengths")
        return self.in_lengths, self.user_inputs, None        


    def get_record_inputs(self):
        return tf.cond(tf.equal(self.train_or_valid, b'train'), 
                       true_fn=lambda : self.train_iter.get_next(), 
                       false_fn=lambda : self.valid_iter.get_next())


    def active_mode(self, mode="train"):
        '''
        @param mode: train | valid
        '''
        self.train_or_valid =  tf.assign(self.train_or_valid, mode)


    def feed_user_dict(self, user_inputs, in_lengths):
        return {self.user_inputs: user_inputs, self.in_lengths: in_lengths}


    def build_pipeline(self, record_file, batch_size, num_epochs, shuffle, scope_name):
        def parse(example_proto):
            context_features = {"seq_length": tf.FixedLenFeature([], dtype=tf.int64)}
            sequence_features = {"chars": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                                 "tags": tf.FixedLenSequenceFeature([], dtype=tf.int64)}
    
            context_parsed, sequence_parsed = tf.parse_single_sequence_example(
                serialized=example_proto,
                context_features=context_features,
                sequence_features=sequence_features
            )
    
            seq_length = context_parsed["seq_length"]
            chars = sequence_parsed["chars"]
            tags = sequence_parsed["tags"]
    
            return seq_length, chars, tags
        
        with tf.variable_scope(scope_name):
            dataset = (tf.contrib.data.TFRecordDataset(record_file).map(parse))
            if shuffle:
                dataset = dataset.shuffle(buffer_size=batch_size*20)
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.padded_batch(batch_size, padded_shapes=([], [None], [None]))
            
            data_iter = dataset.make_initializable_iterator()
            return data_iter.initializer, data_iter
