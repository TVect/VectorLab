import tensorflow as tf
from basic_gan import BasicGAN

tf.logging.set_verbosity(tf.logging.INFO)

tf.app.flags.DEFINE_string("model", "GAN", "model: GAN | LSGAN | WGAN | WGAN-GP")
tf.app.flags.DEFINE_string("mode", "train", "mode: train | infer")
tf.app.flags.DEFINE_integer("noise_dim", 100, "noise dim: default 100")
tf.app.flags.DEFINE_integer("tag_dim", 128, "tag dim")
tf.app.flags.DEFINE_float("learning_rate", 0.0002, "learning rate: default 0.001")
tf.app.flags.DEFINE_float("beta1", 0.5, "Adam beta1")
tf.app.flags.DEFINE_integer("epoches", 1000, "epoches: default 100")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch_size: default 64")

tf.app.flags.DEFINE_integer("d_pretrain", 0, "discriminator pretrain")
tf.app.flags.DEFINE_integer("d_schedule", 15, "train discriminator more ...")

tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint", 
                           "Directory name to save the checkpoints [checkpoint]")
# tf.app.flags.DEFINE_string("data_dir", "./AnimeData_NTU", "Root directory of dataset [data]")
tf.app.flags.DEFINE_string("imgs_dir", "./AnimeData_NTU/extra_data/images", "image directory")
tf.app.flags.DEFINE_string("tags_file", "./AnimeData_NTU/extra_data/tags.csv", "tags file")

tf.app.flags.DEFINE_string("sample_dir", "./sample", "directory of generated images")
tf.app.flags.DEFINE_integer("log_interval", 100, "log interval")
tf.app.flags.DEFINE_integer("max_to_keep", 10, "max to keep")

# WGAN 中权重裁剪的上下限
tf.app.flags.DEFINE_float("clip_min", -0.01, "min value when clip weights, used in WGAN")
tf.app.flags.DEFINE_float("clip_max", 0.01, "max value when clip weights, used in WGAN")

# WGAN-GP 中 gradient_penalty 系数
tf.app.flags.DEFINE_float("penalty_coef", 0.25, "gradient_penalty coef in WGAN-GP")

FLAGS = tf.app.flags.FLAGS


def create_hparams(flags=None):
    """Create training hparams."""
    return tf.contrib.training.HParams(
            model = FLAGS.model,
            noise_dim = FLAGS.noise_dim,
            tag_dim = FLAGS.tag_dim,
            learning_rate = FLAGS.learning_rate,
            beta1 = FLAGS.beta1,
            epoches = FLAGS.epoches,
            batch_size = FLAGS.batch_size,
            d_pretrain = FLAGS.d_pretrain,
            d_schedule = FLAGS.d_schedule,
            
            checkpoint_dir = FLAGS.checkpoint_dir,
            # data_dir = FLAGS.data_dir,
            imgs_dir = FLAGS.imgs_dir,
            tags_file = FLAGS.tags_file,
            sample_dir = FLAGS.sample_dir,
            log_interval = FLAGS.log_interval,
            max_to_keep = FLAGS.max_to_keep,
            
            clip_min = FLAGS.clip_min,
            clip_max = FLAGS.clip_max,
            
            penalty_coef = FLAGS.penalty_coef)


def train():
    hparams = create_hparams(FLAGS)
    gan = BasicGAN(hparams)

    with tf.Session() as sess:
        gan.train(sess)


def infer():
    hparams = create_hparams(FLAGS)
    gan = BasicGAN(hparams)
    
    with tf.Session() as sess:
        gan.infer(sess)


def main(args):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "infer":
        infer()

if __name__ == "__main__":  
    tf.app.run()
