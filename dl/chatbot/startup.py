import jieba
import numpy as np
import tensorflow as tf
from data.cornell.CornellHelper import CornellHelper
from BaseChatModel import BaseChatModel, Config


tf.app.flags.DEFINE_string("mode", "infer", "mode: train | eval | infer")

FLAGS = tf.app.flags.FLAGS

def train():
    data_helper = CornellHelper(max_document_length=50, 
                           filename="data/cornell/dataset-cornell-length10-filter1-vocabSize40000.pkl")
    
    with tf.Graph().as_default():
        config = Config()
        config.vocab_size = data_helper.vocab_size
        config.SOS_TOKEN_ID = data_helper.sosTokenId
        config.EOS_TOKEN_ID = data_helper.eosTokenId

        chat_model = BaseChatModel(config)
        with tf.Session() as session:
            chat_model.train(session, data_helper)


def infer():
    data_helper = CornellHelper(max_document_length=50, 
                           filename="data/cornell/dataset-cornell-length10-filter1-vocabSize40000.pkl")
    
    with tf.Graph().as_default():
        config = Config()
        config.mode = tf.contrib.learn.ModeKeys.INFER
        config.vocab_size = data_helper.vocab_size
        config.SOS_TOKEN_ID = data_helper.sosTokenId
        config.EOS_TOKEN_ID = data_helper.eosTokenId

        chat_model = BaseChatModel(config)
        
        with tf.Session() as session:
            in_sent = "How old are you ?"
            
            parsed_sents = list(tf.contrib.learn.preprocessing.tokenizer([in_sent]))
            import IPython
            IPython.embed()
            en_input, en_length = data_helper.transform(parsed_sents, 
                                                        sos_padding=True, 
                                                        eos_padding=True,
                                                        isTokenId=False)
            preds = chat_model.infer(session, en_input, en_length)
            import IPython
            IPython.embed()
            print("response:", data_helper.re_transform(preds))



def main(args):
    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "infer":
        infer()

if __name__ == "__main__":  
    tf.app.run()
