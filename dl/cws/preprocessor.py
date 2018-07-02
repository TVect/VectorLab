import tqdm
import itertools
import utils
import tensorflow as tf
from MyVocab import MyVocab

class DataPreProcessor:

    def __init__(self, vocab=None):
        if not vocab:
            self.vocab = MyVocab()
        else:
            self.vocab = vocab

    def is_puns(self, c):
        return c in self.vocab.delimiter


    def read_corpus_from_file(self, file_path):
        with open(file_path, encoding='UTF-8') as f:
            for line in f:
                sent = []
                for word in utils.strQ2B(line).strip().split():
                    # TODO 是否需要将数字替换为 #NUM, 将英文替换为 #ENG 
                    sent.append(word)
                    if self.is_puns(word):
                        yield sent
                        sent = []

                if len(sent) > 0:
                    yield sent


    def tag_bmes(self, infile, outfile):
        '''从原始格式转化为 bmes 标注格式'''
        with open(outfile, "w") as fw:
            for words in tqdm.tqdm(self.read_corpus_from_file(infile)):
                for word in words:
                    if len(word) == 1:
                        fw.write("%s\tS\n" % word)
                    else:
                        fw.write("%s\tB\n" % word[0])
                        for idx in range(1, len(word)-1):
                            fw.write("%s\tM\n" % word[idx])
                        fw.write("%s\tE\n" % word[-1])
                fw.write("\n")


    def to_tfrecords(self, infile, outfile, up_vocab=True):
        '''从 bmes 标注格式转化为 tfrecords 格式'''
        writer = tf.python_io.TFRecordWriter(path=outfile)
        chars = []
        tags = []
        sents_cnt = 0
        with open(infile, 'r') as f:
            for line in tqdm.tqdm(f):
                striped_line = line.strip()
                if striped_line:
                    char, tag = striped_line.split()
                    chars.append(char)
                    tags.append(tag)
                    if up_vocab:
                        if char not in self.vocab.chr2id:
                            self.vocab.chr2id[char] = len(self.vocab.chr2id)
                            self.vocab.id2chr.append(char)
                        if tag not in self.vocab.tag2id:
                            self.vocab.tag2id[tag] = len(self.vocab.tag2id)
                            self.vocab.id2tag.append(tag)
                else:
                    sents_cnt += 1
                    writer.write(self.make_example(chars, tags).SerializeToString())
                    chars.clear()
                    tags.clear()
        print("sents cnt: ", sents_cnt)


    def make_example(self, chars, tags):
        # The object we return
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        seq_length = len(chars)
        ex.context.feature["seq_length"].int64_list.value.append(seq_length)
        # Feature lists for the two sequential features of our example
        fl_chars = ex.feature_lists.feature_list["chars"]
        fl_tags = ex.feature_lists.feature_list["tags"]
        for char, tag in zip(chars, tags):
            fl_chars.feature.add().int64_list.value.append(self.vocab.chr2id.get(char, self.vocab.OOV_ID))
            fl_tags.feature.add().int64_list.value.append(self.vocab.tag2id.get(tag))
        return ex


if __name__ == "__main__":
    # trans training file
    print("\n[...... process training file ......]")
    processor = DataPreProcessor()
    infile = "./data/icwb2-data/training/pku_training.utf8"
    bmes_file = "./data/pku_bmes.utf8"
    processor.tag_bmes(infile, bmes_file)

    record_file = "./data/pku.records"
    processor.to_tfrecords(bmes_file, record_file, up_vocab=True)
    processor.vocab.save("myvocab.pkl")
    
    # trans testing file
    print("\n[...... process testing file ......]")
    infile = "./data/icwb2-data/gold/pku_test_gold.utf8"
    bmes_file = "./data/pku_bmes_test.utf8"
    processor.tag_bmes(infile, bmes_file)
    
    record_file = "./data/pku_test.records"
    processor.to_tfrecords(bmes_file, record_file, up_vocab=False)

    import IPython
    IPython.embed()
