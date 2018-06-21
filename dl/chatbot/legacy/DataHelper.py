import os
import pickle
import numpy as np
from tensorflow.contrib import learn
from tensorflow.python.ops import lookup_ops
from MyVocabProcessor import MyVocabProcessor
from W2VVocabProcessor import W2VVocabProcessor
from collections import namedtuple

class DataHelper:
    
    def __init__(self, vocab_processor):
        self.vocab_processor = vocab_processor
        # self.datas 结构为 [([wd1, wd2, ...], [wd1, wd2, ...]), ...]


    def gen_from_rawdata(self, data_obj):
        # TODO 重新调整, 关注 UNK 的处理
        
        self.inputs = np.array(list(data_obj.getInput()))
        self.outputs = np.array(list(data_obj.getOutput()))
        if not os.path.isdir("data/np_data"):
            os.mkdir("data/np_data")
        np.save("data/np_data/data_inputs.npy", self.inputs)
        np.save("data/np_data/data_outputs.npy", self.outputs)

        # self.en_inputs = np.array(list(self.vocab_processor.transform(data_obj.getInput(), sos_padding=True, eos_padding=True)))
        # self.de_inputs = np.array(list(self.vocab_processor.transform(data_obj.getOutput(), sos_padding=True)))
        # self.de_outputs = np.array(list(self.vocab_processor.transform(data_obj.getOutput(), eos_padding=True)))
    
        # self.en_lengths = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in self.en_inputs])
        # self.de_lengths = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in self.de_outputs])
        
#         if not os.path.isdir("data/parsed_data"):
#             os.mkdir("data/parsed_data")
#         np.save("data/parsed_data/en_inputs.npy", self.en_inputs)
#         np.save("data/parsed_data/de_inputs.npy", self.de_inputs)
#         np.save("data/parsed_data/de_outputs.npy", self.de_outputs)
#         np.save("data/parsed_data/en_lengths.npy", self.en_lengths)
#         np.save("data/parsed_data/de_lengths.npy", self.de_lengths)
        

    def gen_from_npy(self, npy_names):
        self.inputs = np.load(npy_names["inputs"])
        self.outputs = np.load(npy_names["outputs"])
        

    def batch_iter(self, epoch=10, batch_size=64):
        data_cnt = len(self.inputs)
        num_batch = int((data_cnt-1)/batch_size)+1

        for epoch_id in range(epoch):
            random_ids = np.random.permutation(data_cnt)
            random_inputs = self.inputs[random_ids]
            random_outputs = self.outputs[random_ids]
            
            for batch_id in range(num_batch):
                start_id = batch_id * batch_size
                end_id = min((batch_id+1)*batch_size, data_cnt)
                batch_en_inputs = np.array(list(self.vocab_processor.transform(random_inputs[start_id: end_id], 
                                                                               sos_padding=True, 
                                                                               eos_padding=True)))
                batch_de_inputs = np.array(list(self.vocab_processor.transform(random_outputs[start_id: end_id], 
                                                                               sos_padding=True, 
                                                                               eos_padding=False)))
                batch_de_outputs = np.array(list(self.vocab_processor.transform(random_outputs[start_id: end_id], 
                                                                                sos_padding=False, 
                                                                                eos_padding=True)))
                batch_en_lengths = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in batch_en_inputs])
                batch_de_lengths = np.array([np.max(np.where(it > 0))+1 if (it != 0).any() else 0 for it in batch_de_inputs])
                
                yield batch_en_inputs, batch_de_inputs, batch_de_outputs, batch_en_lengths, batch_de_lengths


if __name__ == "__main__":
#     vocab_processor = MyVocabProcessor(vcb_file="embedding/wordvecs.vcb", 
#                                        vec_file="embedding/wordvecs.txt",
#                                        max_document_length=40)
    
    vocab_processor = W2VVocabProcessor(w2v_model_file="char2vec.wv", 
                                        max_document_length=100)
    
    data_helper = DataHelper(vocab_processor)
#     from StcWeiboData import StcWeiboData
#     data_helper.gen_from_rawdata(StcWeiboData("data/stc_weibo"))
    
    
    npy_names = {"inputs": "data/np_data/data_inputs.npy",
                 "outputs": "data/np_data/data_outputs.npy"}
    data_helper.gen_from_npy(npy_names)
    
    for item in data_helper.batch_iter():
        import IPython
        IPython.embed()
        break
