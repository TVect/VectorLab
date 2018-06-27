import tqdm
import itertools
import utils

class DataHelper:
    delimiter = set(itertools.chain(u'。！？：；…、，,;!?、,', [u'……']))

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


if __name__ == "__main__":
    infile = "./data/icwb2-data/training/pku_training.utf8"
    outfile = "./pku_bmes.utf8"
    DataHelper.tag_bmes(infile, outfile)

