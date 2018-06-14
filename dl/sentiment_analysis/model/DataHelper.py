import random
import gensim
import keras
import keras.preprocessing.text
import keras.preprocessing.sequence
import numpy as np
from tensorflow.contrib import learn
from sklearn.cross_validation import train_test_split

class DataHelper():
    
    def __init__(self, 
                 training_label_file="../data/training_label.csv", 
                 training_nolabel_file="../data/training_nolabel.csv"):
        self.training_label_file = training_label_file
        self.training_nolabel_file = training_nolabel_file
            
    def read_rawdata(self, filename):
        sentiment_data = []
        sentiment_label = []
        with open(self.training_label_file) as fr:
            for line in fr:
                line_split = line.split("+++$+++")
                if len(line_split) == 2:
                    label = int(line_split[0].strip())
                    sents = line_split[1].strip()
                    # sentence 进行基本的 stem 处理
                    sentiment_data.append(gensim.parsing.preprocessing.stem(sents))
                    sentiment_label.append(label)
        return sentiment_data, sentiment_label

    def build(self):
        sentiment_data, self.sentiment_label = self.read_rawdata(self.training_label_file)
#         self.tokenizer  = keras.preprocessing.text.Tokenizer(num_words=100)
#         self.tokenizer.fit_on_texts(sentiment_data)
#         self.sentiment_vec = self.tokenizer.texts_to_sequences(sentiment_data)
        self.max_length = max(len(sent) for sent in sentiment_data)
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=100, min_frequency=5)
        self.sentiment_vec = np.array(list(self.vocab_processor.fit_transform(sentiment_data)))
        self.sentiment_label = np.array(self.sentiment_label)

    def get_train_test_split(self, test_size=0.05):
        return train_test_split(self.sentiment_vec, self.sentiment_label, test_size=test_size)


if __name__ == "__main__":
    helper = DataHelper()
    helper.build()
    import IPython
    IPython.embed()
