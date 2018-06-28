import tensorflow as tf


tf.app.flags.DEFINE_string("model", "idcnn", "model: idcnn")

FLAGS = tf.app.flags.FLAGS


def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
            vocab_size=10000,
            embed_dim=100,
            num_units=50,
            # dense_units = 50
            keep_ratio=0.5,
            rnn_output_keep_prob=0.5,
            
            # lstm 层数
            num_lstm_layers = 3,
            # 每一个全连接的 units
            dense_units_per_layer = [64, 32], 
            # l2 正则化
            l2_ratio = 1.0,
            # 学习率相关
            lr = 0.001,
            decay_steps = 1000,
            decay_rate = 0.96,
            epoch = 50,
            batch_size = 128
          )



def train():
    pass


def infer():
    pass



def main(args):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "infer":
        infer()

if __name__ == "__main__":  
    tf.app.run()

