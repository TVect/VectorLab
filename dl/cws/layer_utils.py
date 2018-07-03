import numpy as np
import tensorflow as tf


def idcnn_layer(layer_inputs, block_cnt=2, layer_per_block=5, name="idcnn_layer"):
    '''
    @param layer_inputs: [batch_size, seq_length, emb_size]
    @param block_cnt: 有几个block堆叠而成.
    @param layer_per_block: 每个block里面有几个layer.
    @return: [batch_size, seq_length, emb_size]
    '''
    _inputs = tf.expand_dims(layer_inputs, 1)
    channels = _inputs.shape[-1]
    with tf.variable_scope(name):
        for block_id in range(block_cnt):
            for layer_id in range(layer_per_block):
                with tf.variable_scope("conv_layer-{}".format(layer_id), 
                                       reuse=True if block_id > 0 else False):
                    filter_weights = tf.get_variable("w_{}".format(layer_id),
                                                     shape=[1, 3, channels, channels],
                                                     initializer=tf.contrib.layers.xavier_initializer())
                    filter_bias = tf.get_variable("b_{}".format(layer_id), shape=[channels])
                    conv = tf.nn.atrous_conv2d(_inputs, 
                                               filters=filter_weights, 
                                               rate=np.power(2, layer_id), 
                                               padding="SAME", 
                                               name="conv_layer-{}".format(layer_id))
                    conv = tf.nn.bias_add(conv, filter_bias)
                    _inputs = tf.nn.relu(conv)
        return tf.squeeze(_inputs, axis=1, name="output")


def lstm_layer(inputs, in_lengths, lstm_layers, lstm_units, rnn_dropout_ratio=1.0, name="lstm_layer"):
    '''
    @param inputs: [batch_size, seq_length, emb_size]
    @param in_lengths: [batch_size]
    @param lstm_layers: lstm 层数
    @param lstm_units: lstm units 个数
    @param rnn_dropout_ratio: dropout
    @return: [batch_size, seq_length, emb_size]
    '''
    with tf.variable_scope(name):
        def gen_lstm_cell():
            return tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(num_units=lstm_units), 
                                                 output_keep_prob=rnn_dropout_ratio)
    
        # 多层 lstm
        outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [gen_lstm_cell() for _ in range(lstm_layers)], 
            [gen_lstm_cell() for _ in range(lstm_layers)], 
            inputs=inputs, 
            sequence_length=in_lengths,
            dtype=tf.float32)
        return outputs