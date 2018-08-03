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

`opencv-python 3.4.2.17`
