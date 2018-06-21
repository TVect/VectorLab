import os
import numpy as np
import pickle


class CornellHelper:
    
    padTokenId = 0
    sosTokenId = 1
    eosTokenId = 2
    unknownTokenId = 3
    
    def __init__(self, 
                 max_document_length=50, 
                 filename="dataset-cornell-length10-filter1-vocabSize40000.pkl"):
        self.max_document_length = max_document_length
        self.word2id, self.id2word, self.samples = self.load_data(filename)


    @property
    def vocab_size(self):
        return len(self.word2id)
    
    
    def load_data(self, filename):
        '''读取样本数据
        @param filename: 文件路径，是一个字典，包含word2id、id2word分别是单词与索引对应的字典和反序字典，
                       trainingSamples样本数据，每一条都是QA对
        @return: word2id, id2word, trainingSamples
        '''
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            word2id = data['word2id']
            id2word = data['id2word']
            samples = data['trainingSamples']

        return word2id, id2word, np.array(samples)


    def transform(self, samples, eos_padding=False, sos_padding=False, isTokenId=True):
        '''
        @param samples: [[token1, token2,...], [token1, token2, ...], ...]
        @param isTokenId: Boolean, true表示sample中的token是id 
        '''
        word_ids = np.ones([len(samples), self.max_document_length], dtype=np.int32) * self.padTokenId
        word_ids_lenghts = np.zeros(len(samples), dtype=np.int32)

        for s_id, sample in enumerate(samples):
            start_id = 0
            if sos_padding:
                word_ids[s_id][0] = self.sosTokenId
                start_id += 1
            for _id, token in enumerate(sample):
                if _id + start_id >= self.max_document_length:
                    break
                if isTokenId:
                    word_ids[s_id][start_id + _id] = token
                else:
                    word_ids[s_id][start_id + _id] = self.word2id.get(token.lower(), self.unknownTokenId)
            if eos_padding:
                if len(sample) + start_id >= self.max_document_length:
                    word_ids[s_id][-1] = self.eosTokenId
                else:
                    word_ids[s_id][len(sample)+start_id] = self.eosTokenId
            word_ids_lenghts[s_id] = min(start_id + len(sample) + 1, self.max_document_length)
        return word_ids, word_ids_lenghts
    
    
    def re_transform(self, id_sents):
        return [[self.id2word.get(word_id, "UNK") for word_id in word_ids] for word_ids in id_sents]
    
    
    def batch_iter(self, epoch=10, batch_size=64):
        data_cnt = len(self.samples)
        num_batch = int((data_cnt-1)/batch_size)+1

        for epoch_id in range(epoch):
            random_ids = np.random.permutation(data_cnt)
            random_samples = self.samples[random_ids]
            
            for batch_id in range(num_batch):
                start_id = batch_id * batch_size
                end_id = min((batch_id+1)*batch_size, data_cnt)
                
                raw_inputs = random_samples[start_id: end_id][:, 0]
                raw_outputs = random_samples[start_id: end_id][:, 1]

                batch_en_inputs, batch_en_lengths = self.transform(raw_inputs, sos_padding=True, eos_padding=True, isTokenId=True)
                batch_de_inputs, batch_de_lengths = self.transform(raw_outputs, sos_padding=True, eos_padding=False, isTokenId=True)
                batch_de_outputs, _ = self.transform(raw_outputs, sos_padding=False, eos_padding=True, isTokenId=True)

                yield batch_en_inputs, batch_de_inputs, batch_de_outputs, batch_en_lengths, batch_de_lengths


if __name__ == "__main__":
    helper = CornellHelper(max_document_length=50)
    for item in helper.batch_iter():
        import IPython
        IPython.embed()
        break