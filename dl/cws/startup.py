import os
import tensorflow as tf

from BaseModel import BaseModel
from MyVocab import MyVocab
import utils

tf.app.flags.DEFINE_string("model", "idcnn", "model: idcnn")
tf.app.flags.DEFINE_string("mode", "train", "mode: train | eval | infer")

FLAGS = tf.app.flags.FLAGS


vocab = MyVocab.load("myvocab.pkl")

def create_hparams(flags):
  """Create training hparams."""
  return tf.contrib.training.HParams(
            is_user_input=False,     # 接收用户输入或者直接从tfrecord中读取数据
            record_file="./data/pku.records",   # tfrecord文件的位置, 在is_user_input=False时, 必须要填写正确的路径
            train_record="./data/pku_train.records",
            valid_record="./data/pku_valid.records", 
            
            layer_mode = "lstm",   # 采用的layer的类型, idcnn | lstm
            vocab_size=vocab.vocab_size,
            tags_size=vocab.tags_size,

            embed_dim=100,
            
            # lstm 相关参数
            lstm_units=50,      # lstm 中隐藏单元的个数
            lstm_layers = 2,    # lstm 层数
            rnn_output_keep_prob=0.5,

            # idcnn 相关参数
            idcnn_blocks = 3,   # idcnn 中有几个 block
            idcnn_layerpb = 5,  # idcnn 的每个 block 中有几个 layer.

            keep_ratio=0.5,
            l2_ratio = 1.0,     # l2 正则化
            
            # 学习率相关
            lr = 0.001,
            decay_steps = 1000,
            decay_rate = 0.96,
            
            num_epochs = 20,
            batch_size = 64,

            traininfo_every = 10,
            evaluate_every = 100,
            checkpoint_every = 500,
            output_dir = os.path.abspath(os.path.join(os.path.curdir, "model_checkout"))
          )



def train():
    with tf.Graph().as_default():
        hparams = create_hparams(FLAGS)
        cws_model = BaseModel(hparams)
        with tf.Session() as session:
            cws_model.train(session)


def infer():
    with tf.Graph().as_default():
        hparams = create_hparams(FLAGS)
        hparams.is_user_input = True
        cws_model = BaseModel(hparams)
        with tf.Session() as session:
            # in_sent = "中华人民共和国中央人民政府今天成立了。"
            while True:
                in_sent = input("\n====== in_sent: ")
                chars = list(utils.strQ2B(in_sent).strip())
                chars_length = len(chars)
                char_ids = [vocab.chr2id.get(chr, vocab.OOV_ID) for chr in chars]
    
                tags, scores = cws_model.infer(session, [char_ids], [chars_length])
                print("------ score: %s ------" % scores[0])
                for item in zip(chars, tags[0].tolist()):
                    print(item[0], vocab.id2tag[item[1]])




def main(args):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "infer":
        infer()

if __name__ == "__main__":  
    tf.app.run()

