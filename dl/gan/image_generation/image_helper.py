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
        img_list = []

        for img_name in os.listdir(dirname):
            img_file = os.path.join(dirname, img_name)
            img_list.append(np.array(cv2.resize(cv2.imread(img_file, cv2.IMREAD_COLOR), (64, 64)))/255)

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
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        # gen_imgs should be shape (25, 64, 64, 3)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
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
