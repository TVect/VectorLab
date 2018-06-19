import os

"""
stc_weibo 语料
"""

class DataIter:
    def __init__(self, filename):
        self.filename = filename
    
    def __iter__(self):
        with open(self.filename, 'r') as f:
            for line in f:
                yield line.strip()



class StcWeiboData:    

    def __init__(self, dirname):
        """
        Args:
            dirName (string): directory where to load the corpus
        """
        
        self.post_file = os.path.join(dirname, "stc_weibo_train_post")
        self.response_file = os.path.join(dirname, "stc_weibo_train_response")
    
    
    def getInput(self):
        return DataIter(self.post_file)

    def getOutput(self):
        return DataIter(self.response_file)
    
#         self.conversations = list(zip(self.loadLines(self.post_file), self.loadLines(self.response_file)))
# 
# 
#     def loadLines(self, fileName):
#         """
#         Args:
#             fileName (str): file to load
#         """
#         lines = []
#         with open(fileName, 'r') as f:
#             for line in f:
#                 # TODO 对英文词特殊处理
#                 lines.append(line.strip().split())
#         return lines
# 
# 
#     def getConversations(self):
#         return self.conversations




if __name__ == "__main__":
    data = StcWeiboData("data/stc_weibo")
    import IPython
    IPython.embed()
    