import jieba
import numpy as np
from DataHelper import DataHelper
from MyVocabProcessor import MyVocabProcessor
from BaseChatModel import BaseChatModel, Config
import tensorflow as tf

tf.app.flags.DEFINE_string("mode", "infer", "mode: train | eval | infer")

FLAGS = tf.app.flags.FLAGS

def train():
    vocab_processor = MyVocabProcessor(vcb_file="embedding/wordvecs.vcb", 
                                       vec_file="embedding/wordvecs.txt",
                                       max_document_length=40)
    data_helper = DataHelper(vocab_processor)
    
    npy_names = {"en_inputs": "data/parsed_data/en_inputs.npy",
                 "de_inputs": "data/parsed_data/de_inputs.npy", 
                 "de_outputs": "data/parsed_data/de_outputs.npy", 
                 "en_lengths": "data/parsed_data/en_lengths.npy", 
                 "de_lengths": "data/parsed_data/de_lengths.npy"}
    data_helper.gen_from_npy(npy_names)
    
    with tf.Graph().as_default():
        config = Config()
        config.vocab_size = len(data_helper.vocab_processor.vocab_table)
        config.emb_size = data_helper.vocab_processor.vector_size
        config.emb_matrix = data_helper.vocab_processor.get_embedding_matrix()
        config.SOS_TOKEN_ID = data_helper.vocab_processor.SOS_ID
        config.EOS_TOKEN_ID = data_helper.vocab_processor.EOS_ID

        chat_model = BaseChatModel(config)
        
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            chat_model.train(session, data_helper)


def infer():
    vocab_processor = MyVocabProcessor(vcb_file="embedding/wordvecs.vcb", 
                                       vec_file="embedding/wordvecs.txt",
                                       max_document_length=40)
    
    with tf.Graph().as_default():
        config = Config()
        config.mode = tf.contrib.learn.ModeKeys.INFER
        config.vocab_size = len(vocab_processor.vocab_table)
        config.emb_size = vocab_processor.vector_size
        config.emb_matrix = vocab_processor.get_embedding_matrix()
        config.SOS_TOKEN_ID = vocab_processor.SOS_ID
        config.EOS_TOKEN_ID = vocab_processor.EOS_ID

        chat_model = BaseChatModel(config)
        
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            in_sent = "今天的天气真是不错。"
            en_input = np.array(list(vocab_processor.transform([" ".join(jieba.lcut(in_sent))], 
                                                               sos_padding=True, eos_padding=True)))
            en_length = np.array([np.max(np.where(en_input > 0))+1 if (en_input != 0).any() else 0])
            preds = chat_model.infer(session, en_input, en_length)
            import IPython
            IPython.embed()
            print("response:", vocab_processor.re_transform(preds))



def main(args):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "infer":
        infer()

if __name__ == "__main__":  
    tf.app.run()
