import gensim
import tensorflow as tf
from tensorflow.contrib import learn


class SentenceIter:
    def __init__(self):
        from DataHelper import DataHelper
        self.helper = DataHelper()
    
    def __iter__(self):
        labeled_data, _ = self.helper.read_rawdata(self.helper.training_label_file)
        for sent in learn.preprocessing.tokenizer(labeled_data):
            yield sent
        
        nolabeled_data, _ = self.helper.read_rawdata(self.helper.training_nolabel_file)
        for sent in learn.preprocessing.tokenizer(nolabeled_data):
            yield sent


model = gensim.models.Word2Vec(SentenceIter(), size=100, window=5, min_count=5, workers=4)
model.wv.save_word2vec_format("word2vec.stem.wv")

