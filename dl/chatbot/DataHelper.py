import os
import pickle
import numpy as np
from tensorflow.contrib import learn
from tensorflow.python.ops import lookup_ops
from MyVocabProcessor import MyVocabProcessor
from collections import namedtuple

class DataHelper:
    
    def __init__(self, vocab_processor):
        self.vocab_processor = vocab_processor
        # self.datas 结构为 [([wd1, wd2, ...], [wd1, wd2, ...]), ...]


    def gen_from_rawdata(self, data_obj):
        # TODO 重新调整, 关注 UNK 的处理
        self.en_inputs = np.array(list(self.vocab_processor.transform(data_obj.getInput(), sos_padding=True, eos_padding=True)))
        self.de_inputs = np.array(list(self.vocab_processor.transform(data_obj.getOutput(), sos_padding=True)))
        self.de_outputs = np.array(list(self.vocab_processor.transform(data_obj.getOutput(), eos_padding=True)))
    
        self.en_lengths = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in self.en_inputs])
        self.de_lengths = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in self.de_outputs])
        
        if not os.path.isdir("data/parsed_data"):
            os.mkdir("data/parsed_data")
        np.save("data/parsed_data/en_inputs.npy", self.en_inputs)
        np.save("data/parsed_data/de_inputs.npy", self.de_inputs)
        np.save("data/parsed_data/de_outputs.npy", self.de_outputs)
        np.save("data/parsed_data/en_lengths.npy", self.en_lengths)
        np.save("data/parsed_data/de_lengths.npy", self.de_lengths)
        

    def gen_from_npy(self, npy_names):
        self.en_inputs = np.load(npy_names["en_inputs"])
        self.de_inputs = np.load(npy_names["de_inputs"])
        self.de_outputs = np.load(npy_names["de_outputs"])
        self.en_lengths = np.load(npy_names["en_lengths"])
        self.de_lengths = np.load(npy_names["de_lengths"])


    def batch_iter(self, epoch=10, batch_size=64):
        data_cnt = len(self.en_inputs)
        num_batch = int((data_cnt-1)/batch_size)+1

        for epoch_id in range(epoch):
            random_ids = np.random.permutation(data_cnt)
            random_en_inputs = self.en_inputs[random_ids]
            random_de_inputs = self.de_inputs[random_ids]
            random_de_outputs = self.de_outputs[random_ids]
            random_en_lengths = self.en_lengths[random_ids]
            random_de_lengths = self.de_lengths[random_ids]
            
            for batch_id in range(num_batch):
                start_id = batch_id * batch_size
                end_id = min((batch_id+1)*batch_size, data_cnt)
                batch_en_inputs = random_en_inputs[start_id: end_id]
                batch_de_inputs = random_de_inputs[start_id: end_id]
                batch_de_outputs = random_de_outputs[start_id: end_id]
                batch_en_lengths = random_en_lengths[start_id: end_id]
                batch_de_lengths = random_de_lengths[start_id: end_id]
                yield batch_en_inputs, batch_de_inputs, batch_de_outputs, batch_en_lengths, batch_de_lengths


if __name__ == "__main__":
    vocab_processor = MyVocabProcessor(vcb_file="embedding/wordvecs.vcb", 
                                       vec_file="embedding/wordvecs.txt",
                                       max_document_length=40)
    data_helper = DataHelper(vocab_processor)
    from StcWeiboData import StcWeiboData
    data_helper.gen_from_rawdata(StcWeiboData("data/stc_weibo"))
    
    '''
    npy_names = {"en_inputs": "data/parsed_data/en_inputs.npy",
                 "de_inputs": "data/parsed_data/de_inputs.npy", 
                 "de_outputs": "data/parsed_data/de_outputs.npy", 
                 "en_lengths": "data/parsed_data/en_lengths.npy", 
                 "de_lengths": "data/parsed_data/de_lengths.npy"}
    data_helper.gen_from_npy(npy_names)
    '''
    for item in data_helper.batch_iter():
        import IPython
        IPython.embed()
        print(item)
        break
