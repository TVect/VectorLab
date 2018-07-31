import tensorflow as tf

def train():
    pass


def main(args):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "infer":
        infer()

if __name__ == "__main__":  
    tf.app.run()
