import tensorflow as tf
from BaseModel import BaseModel


def create_hparams(flags=None):
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
            tags_size = 4,
            # l2 正则化
            l2_ratio = 1.0,
            # 学习率相关
            lr = 0.001,
            decay_steps = 1000,
            decay_rate = 0.96,
            epoch = 50,
            batch_size = 128)

hparams = create_hparams()

base_model = BaseModel(hparams)
