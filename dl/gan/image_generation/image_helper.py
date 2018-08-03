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
    
    def iter_images(self, dirname="./AnimeData_NTU/AnimeData/faces", batch_size=25, epoches=10):
        '''每次返回一个 batch_size 的images, images 归一化到了[-1, 1]'''
        img_list = []

        for img_name in os.listdir(dirname):
            img_file = os.path.join(dirname, img_name)
            # normalize the images between -1 and 1
            img_list.append(np.array(cv2.resize(cv2.imread(img_file, cv2.IMREAD_COLOR), (64, 64)))/127.5 - 1)

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
        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("{}.png".format(img_name))
        plt.close()


if __name__ == "__main__":
    img_helper = ImageHelper()
    for batch_images in img_helper.iter_images():
        print(batch_images)
        import IPython
        IPython.embed()
        exit(0)
