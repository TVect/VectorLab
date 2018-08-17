import tensorflow as tf

class CondEncoder:
    
    def __init__(self, hparams, name="cond_encoder"):
        self.hparams = hparams
        self.name = name if name else "cond_encoder"
        self._build()


    def _build(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.tag_weight = tf.get_variable(name="tag_weight", shape=[22, self.hparams.tag_dim], 
                                              initializer=tf.contrib.layers.xavier_initializer())
            self.tag_bias = tf.get_variable(name="tag_bias", shape=[self.hparams.tag_dim],
                                            initializer=tf.zeros_initializer())
    
    
    def get_representation(self, input):
        return tf.nn.xw_plus_b(input, self.tag_weight, self.tag_bias)
