
# Tips

## ganhacks tips

实现中使用了 [ganhacks](https://github.com/soumith/ganhacks) 中提到的一些Tips:

**1: Normalize the inputs**

- [x] normalize the images between -1 and 1
- [x] Tanh as the last layer of the generator output

**3: Use a spherical Z**

- [x] Dont sample from a Uniform distribution. Sample from a gaussian distribution

**4: BatchNorm**

- [x] when batchnorm is not an option use instance normalization.

**5: Avoid Sparse Gradients: ReLU, MaxPool**

- [x] LeakyReLU = good (in both G and D)

**9: Use the ADAM Optimizer**

- [x] optim.Adam rules!

## DCGAN tips

DCGAN的文章中提到了一些tips, 对实现很有帮助

Architecture guidelines for stable Deep Convolutional GANs

- [x] Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).

- [x] Use batchnorm in both the generator and the discriminator.

- [x] Remove fully connected hidden layers for deeper architectures.

- [x] Use ReLU activation in generator for all layers except for the output, which uses Tanh.

- [x] Use LeakyReLU activation in the discriminator for all layers

---

# 其他

1. AdamOptimizer 中  beta1 在训练中很重要.

之前使用的 AdamOptimizer(learning_rate=0.0002), beta1 取的是默认值 0.9. 训练的结果很糊. 开始训练不久, discriminator 准确率基本就稳定在 1.0 附近了, 再训练到后面 generator 就失效了, 产生了纯色?

后面看到别人代码中一般都将beta1设置为0.5, 尝试了一下, 有效果, discriminator 不再是立即稳定在准确率1.0附近了. 

对了, DCGAN的文章中也有提到. > leaving the momentum term β1 at the suggested value of 0.9 resulted in training oscillation and instability while reducing it to 0.5 helped stabilize training.

2. 修改为 WGAN-GP 之后, 并没有再出现前面提到的训练到很后面 generator 失效, 产生纯色的问题.

---

# 结果展示

![WGAN-GP 生成的图片](./infer-image.png)

![脸像检测的结果](./baseline_result.png)

---

# 参考资料

- [各种 GAN 的代码实现 tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)

- [代码 DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)

---

`opencv-python 3.4.2.17`
