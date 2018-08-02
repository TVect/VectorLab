import tensorflow as tf

# leaky relu
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    # 本质其实就是做了一个matmul....
    shape = input_.get_shape().as_list()
    
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", 
                                 [shape[1], output_size], 
                                 tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", 
                               [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias