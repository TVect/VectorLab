参考了 [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow) 的实现

# Tips
实现中使用了 [ganhacks](https://github.com/soumith/ganhacks) 中提到的一些Tips:

**1: Normalize the inputs**

>- [x] normalize the images between -1 and 1
>- [x] Tanh as the last layer of the generator output

**3: Use a spherical Z**

>- [x] Dont sample from a Uniform distribution. Sample from a gaussian distribution

**4: BatchNorm**

>- [x] when batchnorm is not an option use instance normalization.

**5: Avoid Sparse Gradients: ReLU, MaxPool**

>- [x] LeakyReLU = good (in both G and D)

**9: Use the ADAM Optimizer**

>- [x] optim.Adam rules!

---
# 总结

1. AdamOptimizer 中  beta1 在训练中很重要.

之前使用的 AdamOptimizer(learning_rate=0.0002), beta1 取的是默认值 0.9. 训练的结果很糊. 开始训练不久, discriminator 准确率基本就稳定在 1.0 附近了, 再训练到后面 generator 就失效了, 产生了纯色?

后面看到别人代码中一般都将beta1设置为0.5, 尝试了一下, 有效果, discriminator 不再是立即稳定在准确率1.0附近了.

---

`opencv-python 3.4.2.17`
