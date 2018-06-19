import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

class W2VVocabProcessor:
    ''' 从  gensim.model.Word2Vec 构造Vocabulary.
    Tips: 在Word2Vec中没有 <UNK>, 这里会将<UNK>放到index=0的位置, 其他word相应后移
    '''
    
    def __init__(self, w2v_model_file, max_document_length, tokenizer_fn=None):
        self.w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_file)
        self.max_document_length = max_document_length

        if tokenizer_fn:
            self._tokenizer = tokenizer_fn
        else:
            self._tokenizer = learn.preprocessing.tokenizer


    def transform(self, raw_documents, eos_padding=False, sos_padding=False):
        """Transform documents to word-id matrix.
        
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        
        Args:
          raw_documents: An iterable which yield either str or unicode.
        
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.word2id(token)
            yield word_ids


    def word2id(self, wd):
        vocab = self.w2v_model.vocab.get(wd)
        if vocab:
            return vocab.index + 1
        else:
            return 0


    def get_embedding_matrix(self):
        return np.concatenate(np.zeros([1, self.w2c_model.vector_size]), self.w2c_model.syn0)

