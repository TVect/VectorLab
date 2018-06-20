import numpy as np
from tensorflow.contrib import learn

class MyVocabProcessor:
    
    UNK = "#OOV#"
    SOS = "<a>"
    EOS = "</a>"
    UNK_ID = 0
    SOS_ID = 1
    EOS_ID = 2

    def __init__(self, vcb_file, vec_file, max_document_length, tokenizer_fn=None):
        '''
        @param vcb_file: 词表, 起始的字符为 #OOV#, <a>, </a>
        @param vec_file: 对应的词向量
        '''
        self.max_document_length = max_document_length

        if tokenizer_fn:
            self._tokenizer = tokenizer_fn
        else:
            self._tokenizer = learn.preprocessing.tokenizer

        self.index2word, self.vocab_table, self.vec_rep = self.build_vocab(vcb_file, vec_file)


    def build_vocab(self, vcb_file, vec_file):
        vocab_table = {}
        index2word = []
        with open(vcb_file) as fr:
            for item in fr:
                if item.strip() in vocab_table:
                    continue
                vocab_table[item.strip()]  = len(vocab_table)
                index2word.append(item.strip())

        with open(vec_file) as fr:
            vec_rep = np.array([list(map(float, line.split(","))) for line in fr])
    
        return index2word, vocab_table, vec_rep


    def word2id(self, token):
        word_id = self.vocab_table.get(token, -1)
        if word_id == -1:
            return self.UNK_ID
        return word_id

    
    @property
    def vector_size(self):
        return self.vec_rep.shape[1]
    

    def transform(self, raw_documents, sos_padding=False, eos_padding=False):
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
                word_ids[idx + start_id + 1] = self.EOS_ID
            yield word_ids


    def re_transform(self, id_sents):
        '''
        [[id1, id2, id3, ...], [id4, id5, ...],...] -> [[wd1, wd2, wd3, ...], [wd3, wd4, ...],...]
        '''
        return [[self.index2word[word_id] for word_id in word_ids] for word_ids in id_sents]


    def get_embedding_matrix(self):
        return self.vec_rep
    


if __name__ == "__main__":
    vocab_processor = MyVocabProcessor(vcb_file="embedding/wordvecs.vcb", 
                                       vec_file="embedding/wordvecs.txt",
                                       max_document_length=40)
    import IPython
    IPython.embed()
