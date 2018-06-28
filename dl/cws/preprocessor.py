import tqdm
import itertools
import utils
import tensorflow as tf

class DataPreProcessor:

    delimiter = set(itertools.chain(u'。！？：；…、，,;!?、,', [u'……']))
    
    PAD_ID = 0
    OOV_ID = 1
    PAD_CHR = "##PAD##"
    OOV_CHR = "##OOV##"

    chr2id = {OOV_CHR: OOV_ID}
    id2chr = [OOV_ID]
    tag2id = {}
    id2tag = []

    @classmethod
    def is_puns(cls, c):
        return c in cls.delimiter


    @classmethod
    def read_corpus_from_file(cls, file_path):
        with open(file_path) as f:
            for line in f:
                sent = []
                for word in utils.strQ2B(line).strip().split():
                    # TODO 是否需要将数字替换为 #NUM, 将英文替换为 #ENG 
                    sent.append(word)
                    if cls.is_puns(word):
                        yield sent
                        sent = []

                if len(sent) > 0:
                    yield sent


    @classmethod
    def tag_bmes(cls, infile, outfile):
        '''从原始格式转化为 bmes 标注格式'''
        with open(outfile, "w") as fw:
            for words in tqdm.tqdm(cls.read_corpus_from_file(infile)):
                for word in words:
                    if len(word) == 1:
                        fw.write("%s\tS\n" % word)
                    else:
                        fw.write("%s\tB\n" % word[0])
                        for idx in range(1, len(word)-1):
                            fw.write("%s\tM\n" % word[idx])
                        fw.write("%s\tE\n" % word[-1])
                fw.write("\n")


    @classmethod
    def to_tfrecords(cls, infile, outfile):
        '''从 bmes 标注格式转化为 tfrecords 格式'''
        writer = tf.python_io.TFRecordWriter(path=outfile)
        chars = []
        tags = []
        with open(infile, 'r') as f:
            for line in tqdm.tqdm(f):
                striped_line = line.strip()
                if striped_line:
                    char, tag = striped_line.split()
                    chars.append(char)
                    tags.append(tag)
                    if char not in cls.chr2id:
                        cls.chr2id[char] = len(cls.chr2id)
                        cls.id2chr.append(char)
                    if tag not in cls.tag2id:
                        cls.tag2id[tag] = len(cls.tag2id)
                        cls.id2tag.append(tag)

                else:
                    writer.write(cls.make_example(chars, tags).SerializeToString())
                    chars.clear()
                    tags.clear()


    @classmethod
    def make_example(cls, chars, tags):
        # The object we return
        ex = tf.train.SequenceExample()
        # A non-sequential feature of our example
        seq_length = len(chars)
        ex.context.feature["seq_length"].int64_list.value.append(seq_length)
        # Feature lists for the two sequential features of our example
        fl_chars = ex.feature_lists.feature_list["chars"]
        fl_tags = ex.feature_lists.feature_list["tags"]
        for char, tag in zip(chars, tags):
            fl_chars.feature.add().int64_list.value.append(cls.chr2id.get(char, cls.OOV_ID))
            fl_tags.feature.add().int64_list.value.append(cls.tag2id.get(tag))
        return ex


if __name__ == "__main__":
    infile = "./data/icwb2-data/training/pku_training.utf8"
    bmes_file = "./pku_bmes.utf8"
    # DataHelper.tag_bmes(infile, bmes_file)

    record_file = "./data/pku.records"
    DataHelper.to_tfrecords(bmes_file, record_file)
