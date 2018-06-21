import gensim
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

class W2VVocabProcessor:
    ''' 从  gensim.model.Word2Vec 构造Vocabulary.
    Tips: 在Word2Vec中没有 <UNK>, 这里会将<UNK>放到index=0的位置, 其他word相应后移
    '''
    
    UNK = "#OOV#"
    SOS = "<a>"
    EOS = "</a>"
    UNK_ID = 0
    SOS_ID = 1
    EOS_ID = 2

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
            # 在需要做 sos padding 时, start_id=1, 否认为 0
            start_id = 0
            if sos_padding:
                word_ids[0] = self.SOS_ID
                start_id = 1
            for idx, token in enumerate(tokens):
                if idx + start_id >= self.max_document_length:
                    break
                word_ids[idx + start_id] = self.word2id(token)
            if eos_padding:
                word_ids[len(tokens) + start_id] = self.EOS_ID
            yield word_ids


    def word2id(self, wd):
        wd_id = self.vocab_table.get(wd, -1)
        if wd_id == -1:
            return self.UNK_ID
        return wd_id

    
    def id2word(self, wd_id):
        if wd_id == self.UNK_ID:
            return self.UNK
        else:
            return self.w2v_model.index2word[wd_id-1]


    def re_transform(self, id_sents):
        '''
        [[id1, id2, id3, ...], [id4, id5, ...],...] -> [[wd1, wd2, wd3, ...], [wd3, wd4, ...],...]
        '''
        return [[self.id2word(word_id) for word_id in word_ids] for word_ids in id_sents]


    def get_embedding_matrix(self):
        return np.concatenate((np.zeros([1, self.w2v_model.vector_size]), self.w2v_model.syn0))
    

