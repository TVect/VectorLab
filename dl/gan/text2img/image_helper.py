'''
image helper
'''

import os
import cv2
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from conda._vendor.auxlib._vendor.five import string

plt.switch_backend('agg')


class ImageHelper:
    
    def __init__(self):
        self.hair_tags = ['purple hair', 'red hair', 'brown hair', 'white hair', 
                          'orange hair', 'black hair', 'blue hair', 'pink hair', 
                          'blonde hair', 'aqua hair', 'green hair', 'gray hair']
        self.eyes_tags = ['black eyes', 'pink eyes', 'orange eyes', 'purple eyes', 
                          'aqua eyes', 'red eyes', 'blue eyes', 'green eyes', 
                          'yellow eyes', 'brown eyes']
        self.hair_tag2id = dict([(tag, id) for id, tag in enumerate(self.hair_tags)])
        self.eyes_tag2id = dict([(tag, id) for id, tag in enumerate(self.eyes_tags)])
        self.hair_size = len(self.hair_tags)
        self.eyes_size = len(self.eyes_tags)
        self.tag_dim = self.hair_size + self.eyes_size

        self.patterns = re.compile("(.+ hair) (.+ eyes)")

    def iter_images(self, 
                    imgs_dir="./AnimeData_NTU/extra_data/images", 
                    tags_file="./AnimeData_NTU/extra_data/tags.csv",
                    batch_size=25, 
                    epoches=10):
        '''每次返回一个 batch_size 的images, images 归一化到了[-1, 1]'''
        img_list = []
        tag_list = []
        img_tags = {}
        
        with open(tags_file) as fr:
            for line in fr:
                id, des = line.split(",")
                img_tags[id] = self.patterns.match(des).groups()

        for root, dirs, files in os.walk(imgs_dir):
            for name in files:
                if name.endswith(".jpg"):
                    img_file = os.path.join(root, name)
                    img = cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    # normalize the images between -1 and 1
                    img_list.append(np.array(cv2.resize(img, (64, 64)))/127.5 - 1)
                    # tag_list.append(self.tags2vec(img_tags[name.split(".")[0]]))
                    tag = img_tags[name.split(".")[0]]
                    tag_list.append(self.tag2id(tag))

        img_array = np.array(img_list)
        tag_array = np.array(tag_list)

        del img_list
        del tag_list
        img_shape = img_array.shape
        for num_epoch in range(epoches):
            # batch_per_epoch = int(np.ceil(img_shape[0] / batch_size))
            batch_per_epoch = int(img_shape[0] / batch_size)
            random_ids = np.random.permutation(img_shape[0])
            random_img_array = img_array[random_ids]
            random_tag_array = tag_array[random_ids]
            for num_batch in range(batch_per_epoch):
                batch_imgs = random_img_array[num_batch*batch_size : (num_batch+1)*batch_size]
                # 构造 tags 的 one-hot 表示
                batch_tags = np.zeros([batch_size, self.tag_dim])
                # 构造  wrong tags 的 one-hot 表示
                batch_wtags = np.zeros([batch_size, self.tag_dim])
                for idx, tag in enumerate(random_tag_array[num_batch*batch_size : (num_batch+1)*batch_size]):
                    id1, id2 = self.gen_wtag(tag)
                    batch_wtags[idx][id1] = 1
                    batch_wtags[idx][id2+self.hair_size] = 1
                    batch_tags[idx][tag[0]] = 1
                    batch_tags[idx][tag[1]+self.hair_size] = 1
                yield (num_epoch, num_batch), (batch_imgs, batch_tags, batch_wtags)


    def get_test_tags(self):
        str_tags = ["blue hair blue eyes", "blue hair blue eyes", "blue hair blue eyes",
                    "blue hair blue eyes", "blue hair blue eyes", "blue hair green eyes",
                    "blue hair green eyes", "blue hair green eyes", "blue hair green eyes",
                    "blue hair green eyes", "blue hair red eyes", "blue hair red eyes",
                    "blue hair red eyes", "blue hair red eyes", "blue hair red eyes",
                    "green hair blue eyes", "green hair blue eyes", "green hair blue eyes",
                    "green hair blue eyes", "green hair blue eyes", "green hair red eyes",
                    "green hair red eyes", "green hair red eyes", "green hair red eyes",
                    "green hair red eyes"]
        batch_tags = np.zeros([len(str_tags), self.tag_dim])
        for idx, tag in enumerate(str_tags):
            tag_ids = self.tag2id(self.patterns.match(tag).groups())
            batch_tags[idx][tag_ids[0]] = 1
            batch_tags[idx][tag_ids[1]+self.hair_size] = 1
        return batch_tags


    def tag2id(self, tag):
        '''e.g. tag = ['purple hair', 'black eyes']  -> [0, 0]'''
        return [self.hair_tag2id[tag[0]], self.eyes_tag2id[tag[1]]]

    def gen_wtag(self, tag):
        '''产生 wrong tag'''
        hair_id = random.randint(0, self.hair_size-2)
        if hair_id >= tag[0]: hair_id -= 1
        eyes_id = random.randint(0, self.eyes_size-2)
        if eyes_id >= tag[0]: eyes_id -= 1
        return [hair_id, eyes_id]


    def tags2vec(self, tags):
        vec = np.zeros(len(self.tags))
        for tag in tags:
            vec[self.tag2id.get(tag)] = 1
        return vec


    def save_imgs(self, imgs, img_name):
        '''保存图片, 输入的 imgs 数值在 [-1, 1] 之间, 会在做保存之前做变换'''
        # imgs should be shape (25, 64, 64, 3)
        imgs = (imgs + 1.0)/2.0
        img_shape = imgs.shape
        row_cnt = int(np.sqrt(img_shape[0]))
        col_cnt = int(np.ceil(img_shape[0]/row_cnt))
        fig, axs = plt.subplots(row_cnt, col_cnt)
        for img_id in range(img_shape[0]):
            row_id = int(img_id / col_cnt)
            col_id = img_id % col_cnt
            axs[row_id, col_id].imshow(imgs[img_id, :,:,:])
            axs[row_id, col_id].axis('off')

        fig.savefig("{}.png".format(img_name))
        plt.close()


if __name__ == "__main__":
    img_helper = ImageHelper()
    
    tags = img_helper.get_test_tags()
    import IPython
    IPython.embed()
#     for batch_data in img_helper.iter_images():
#         print(batch_data)
#         import IPython
#         IPython.embed()
#         exit(0)
