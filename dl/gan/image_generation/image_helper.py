'''
image helper
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


class ImageHelper:
    
    def __init__(self):
        pass
    
    def iter_images(self, dirname="./AnimeData_NTU", batch_size=25, epoches=10):
        '''每次返回一个 batch_size 的images, images 归一化到了[-1, 1]'''
        img_list = []

        for root, dirs, files in os.walk(dirname):
            for name in files:
                if name.endswith(".jpg"):
                    img_file = os.path.join(root, name)
                    img = cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    # normalize the images between -1 and 1
                    img_list.append(np.array(cv2.resize(img, (64, 64)))/127.5 - 1)
        '''
        for img_name in os.listdir(dirname):
            img_file = os.path.join(dirname, img_name)
            img = cv2.cvtColor(cv2.imread(img_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # normalize the images between -1 and 1
            img_list.append(np.array(cv2.resize(img, (64, 64)))/127.5 - 1)
        '''
        img_array = np.array(img_list)
        del img_list
        img_shape = img_array.shape
        for num_epoch in range(epoches):
            # batch_per_epoch = int(np.ceil(img_shape[0] / batch_size))
            batch_per_epoch = int(img_shape[0] / batch_size)
            random_img_array = img_array[np.random.permutation(img_shape[0])]
            for num_batch in range(batch_per_epoch):
                yield num_epoch, num_batch, random_img_array[num_batch*batch_size : (num_batch+1)*batch_size]


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
    for batch_images in img_helper.iter_images():
        print(batch_images)
        import IPython
        IPython.embed()
        exit(0)
