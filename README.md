# WAN

**Note (1):** The WAN method has been greatly improved by selecting **compactly supported RBFs (CSRBFs)** as test functions. One can refer to our recent work **[ParticleWNN](https://arxiv.org/pdf/2305.12433)** and its implementation through link: [https://github.com/yaohua32/ParticleWNN](https://github.com/yaohua32/ParticleWNN) or [https://github.com/yaohua32/Physics-Driven-Deep-Learning-for-PDEs](https://github.com/yaohua32/Physics-Driven-Deep-Learning-for-PDEs) where comparison between ParticleWNN and other methods, such as PINN and DeepRitz, on several benchmark PDE problems is provided.

**Note (2):** We have also developed a novel deep neural operator method for solving parametric PDEs and related inverse problems based on the ParticleWNN method, named **DGenNO**. This is the first deep neural operator method based on the **weak form** for solving PDEs **without any labeled training pairs**. One can refer to the work **[DGenNO](https://www.sciencedirect.com/science/article/pii/S0021999125004206)** and its implementation through the link: [https://github.com/yaohua32/Deep-Neural-Operators-for-PDEs](https://github.com/yaohua32/Deep-Neural-Operators-for-PDEs), where a comparison between DGenNO and other DNOs, such as DeepONet, FNO, PI-DeepONet, and PINO, on several benchmark PDE problems is provided.

## A brief introduction to WAN
Weak Adversarial Networks (WAN) method is a deep learning based method for solving high dimensional partial differential equations. It is a mesh-free method which bases on the weak form of solutions for PDEs. In brief, it converts the problem of finding the weak solution of PDEs into an operator norm minimization problem induced from the weak formulation. Then, a GAN-like algorithm was designed for solving this problem by parameterizing the trial function and the test function as the primal and adversarial networks respectively.

## Paper
This repository contains the code for examples in the paper [Weak Adversarial Networks for High-dimensional Partial Differential Equations](https://arxiv.org/abs/1907.08272) by Yaohua Zang, Gang Bao, Xiaojing Ye and Haomin Zhou.

## About the code
* The code in this repository was written by 'python 3.6' and [Tensorflow1](https://www.tensorflow.org/).
* One should keep it in mind that the parameters provided in this code may not efficiently work for different types of problems. So one may need readjust parameters when use this code for solving different problems.
* In recent experiments, we found that several modifications can improve the efficiency of the algorithm. For example, using the [ResNet](https://arxiv.org/abs/1512.03385) for parameterizing the trial function and test function rather than using the Fully Connected Feedforward Network and optimizing the primal network with the 'Adam' optimizer.
