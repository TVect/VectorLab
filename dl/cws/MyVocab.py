import pickle
import itertools

class MyVocab:
    
    delimiter = set(itertools.chain(u'。！？：；…、，,;!?、,', [u'……']))
        
    PAD_ID = 0
    OOV_ID = 1
    PAD_CHR = "##PAD##"
    OOV_CHR = "##OOV##"

    
    def __init__(self):
    
        self.chr2id = {self.OOV_CHR: self.OOV_ID}
        self.id2chr = [self.OOV_ID]
        self.tag2id = {}
        self.id2tag = []


    @property
    def vocab_size(self):
        return len(self.id2chr)


    @property
    def tags_size(self):
        return len(self.id2tag)
    
    
    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as fr:
            data_dict = pickle.load(fr)
        vocab = cls()
        vocab.chr2id = data_dict["chr2id"]
        vocab.id2chr = data_dict["id2chr"]
        vocab.tag2id = data_dict["tag2id"]
        vocab.id2tag = data_dict["id2tag"]
        return vocab


    def save(self, filename):
        with open(filename, "wb") as fw:
            pickle.dump({"chr2id": self.chr2id,
                         "id2chr": self.id2chr,
                         "tag2id": self.tag2id,
                         "id2tag": self.id2tag}, fw)


if __name__ == "__main__":
    vocab = MyVocab.load(filename="myvocab.pkl")
    import IPython
    IPython.embed()

        