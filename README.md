# MNIST_Classifier
Digit recognition with multilayer perceptron classifier using PyTorch 

## Contents
1. Objective
2. Dataset


## 1. Objective
This project aims to develop a **handwritten digit classifier** on [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## 2. Dataset
MNIST contains 70,000 grayscale images (28x28 pixels) of handwritten digits: 60,000 for training and 10,000 for testing. 
###  2.1 Requirments 
[pytorch](https://pytorch.org/docs/stable/index.html) is an open source machine learning framework that provides an easy implementation to download the cleaned and already prepared data.
- **Torch** - the [torch](https://pytorch.org/docs/stable/torch.html) package contains data structures for multi-dimensional tensors and defines mathematical operations over these tensors.
- **TorchVision** - the [torchvision](https://pytorch.org/vision/stable/index.html#torchvision) package (part of the PyTorch project) consists of popular datasets, model architectures, and common image transformations for computer vision.

```
import torch
import torchvision 
```
