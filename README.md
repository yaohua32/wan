# WAN

## A brief introduction to WAN
Weak Adversarial Networks (WAN) method is a deep learning based method for solving high dimensional partial differential equations. It is a mesh-free method which bases on the weak form of solutions for PDEs. In brief, it converts the problem of finding the weak solution of PDEs into an operator norm minimization problem induced from the weak formulation. Then, a GAN-like algorithm was designed for solving this problem by parameterizing the trial function and the test function as the primal and adversarial networks respectively.

## Paper
This repository contains the code for examples in the paper [Weak Adversarial Networks for High-dimensional Partial Differential Equations](https://arxiv.org/abs/1907.08272) by Yaohua Zang, Gang Bao, Xiaojing Ye and Haomin Zhou.

## About the code
* The code in this repository was written by 'python 3.6' and [Tensorflow1](https://www.tensorflow.org/).
* One should keep it in mind that the parameters provided in this code may not efficiently work for different types of problems. So one may need readjust parameters when use this code for solving different problems.
* In recent experiments, we found that several modifications can improve the efficiency of the algorithm. For example, using the [ResNet](https://arxiv.org/abs/1512.03385) for parameterizing the trial function and test function rather than using the Fully Connected Feedforward Network and optimizing the primal network with the 'Adam' optimizer.
