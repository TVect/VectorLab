import tensorflow as tf
from basic_gan import BasicGAN

tf.app.flags.DEFINE_string("model", "LSGAN", "model: LSGAN | WGAN")
tf.app.flags.DEFINE_string("mode", "train", "mode: train | infer")
tf.app.flags.DEFINE_integer("noise_dim", 100, "noise dim: default 100")
tf.app.flags.DEFINE_float("learning_rate", 0.0002, "learning rate: default 0.001")
tf.app.flags.DEFINE_integer("epoches", 1000, "epoches: default 100")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch_size: default 64")

FLAGS = tf.app.flags.FLAGS

def create_hparams(flags=None):
    """Create training hparams."""
    return tf.contrib.training.HParams(
            noise_dim = FLAGS.noise_dim,
            learning_rate = FLAGS.learning_rate,
            epoches = FLAGS.epoches,
            batch_size = FLAGS.batch_size)


def train():
    hparams = create_hparams(FLAGS)
    gan = BasicGAN(hparams)
    gan.train()


def infer():
    pass


def main(args):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "infer":
        infer()

if __name__ == "__main__":  
    tf.app.run()
