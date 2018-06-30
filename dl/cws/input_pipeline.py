import tensorflow as tf

class InputPipeline:
    
    def __init__(self, is_user_input=False, record_file=None):
        self.is_user_input = is_user_input
        if not self.is_user_input:
            assert record_file is not None
            self.batch_data = self.build_pipeline(record_file)
        
    
    def get_inputs(self):
        if self.is_user_input:
            self.user_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="user_inputs")
            self.in_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name="in_lengths")
            return self.user_inputs, self.in_lengths, None
        else:
            return tf.cast(self.batch_data["inputs"], tf.int32), \
                tf.cast(self.batch_data["in_lengths"], tf.int32), \
                tf.cast(self.batch_data["labels"], tf.int32)
    

    def feed_user_dict(self, user_inputs, in_lengths):
        return {self.user_inputs: user_inputs, self.in_lengths: in_lengths}


    def build_pipeline(self, record_file):
        file_queue = tf.train.string_input_producer([record_file], num_epochs=10)
        reader = tf.TFRecordReader()
        _, serialized_example  = reader.read(file_queue)
        
        context_features = {"seq_length": tf.FixedLenFeature([], dtype=tf.int64)}
        sequence_features = {"chars": tf.FixedLenSequenceFeature([], dtype=tf.int64),
                             "tags": tf.FixedLenSequenceFeature([], dtype=tf.int64)}
        
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized=serialized_example,
            context_features=context_features,
            sequence_features=sequence_features
        )
        
        # TODO 构造shuffle的数据
        batch_data = tf.train.batch({"inputs": sequence_parsed["chars"],
                                     "labels": sequence_parsed["tags"],
                                     "in_lengths": context_parsed["seq_length"]},
                                    batch_size=32,
                                    capacity=32*20,
                                    dynamic_pad=True,
                                    allow_smaller_final_batch=True)
        return batch_data