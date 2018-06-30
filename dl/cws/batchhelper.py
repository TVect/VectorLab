import tensorflow as tf


record_file = "./data/pku.records"
shuffle = True

'''
example = next(tf.python_io.tf_record_iterator(record_file))
# tf.train.Example.FromString(example)
import IPython
IPython.embed()
'''


class BatchHelper:
    
    def __init__(self, record_file):
        self.record_file = record_file
        self.batch_data = self.build()

    def build(self):
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
        
        seq_length = context_parsed["seq_length"]
        chars = sequence_parsed["chars"]
        tags = sequence_parsed["tags"]
        
        batch_data = tf.train.batch([seq_length, chars, tags],
                                    batch_size=32,
                                    capacity=64,
                                    dynamic_pad=True,
                                    allow_smaller_final_batch=False)
        return batch_data


    def batch_iter(self, sess):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                rets = sess.run(self.batch_data)
                print(len(rets))
        except:
            coord.request_stop()
        finally:
            coord.join(threads)


if __name__ == "__main__":
    record_file = "./data/pku.records"
    helper = BatchHelper(record_file)
    with tf.Session() as sess:
        helper.batch_iter(sess)