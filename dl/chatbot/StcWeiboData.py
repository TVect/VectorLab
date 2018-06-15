import os

"""
stc_weibo 语料
"""

class StcWeiboData:

    def __init__(self, dirName):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        
        self.post_file = os.path.join(dirName, "stc_weibo_train_post")
        self.response_file = os.path.join(dirName, "stc_weibo_train_response")
        
        self.conversations = list(zip(self.loadLines(self.post_file), self.loadLines(self.response_file)))


    def loadLines(self, fileName):
        """
        Args:
            fileName (str): file to load
        """
        lines = []
        with open(fileName, 'r') as f:
            for line in f:
                # TODO 对英文词特殊处理
                lines.append(line.strip().split())
        return lines


    def getConversations(self):
        return self.conversations

if __name__ == "__main__":
    data = StcWeiboData("data/stc_weibo")
    import IPython
    IPython.embed()
    