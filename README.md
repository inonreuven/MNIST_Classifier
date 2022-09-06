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
[transforms](https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html) provides manipulation of the data to suit it for training.  
- **ToTensor** - a torch.Tensor is a multi-dimensional matrix containing elements of a single data type. [transforms.ToTensor](https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html) Convert a PIL Image or numpy.ndarray to tensor.
- **Normalize** - normalize a tensor image with mean and standard deviation.
- **Compose** - composes several transforms together.

```
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ]) 
```

###  2.2 load MNIST dataset
[datasets](https://pytorch.org/vision/stable/datasets.html) provides the MNIST built-in dataset:
- **'PATH_TO_STORE_X'** - the root directory of dataset.
- **download** - downloads the dataset from the internet and puts it in root directory.  
- **train** - creates the dataset from  train-images-idx3-ubyte/t10k-images-idx3-ubyte. 

[DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) combines a dataset and a sampler, and provides an iterable over the given dataset:
- **dataset** – dataset from which to load the data.
- **batch_size** – how many samples per batch to load.
- **shuffle** – reshuffled the data every epoch.

```
from torchvision import transforms, datasets

trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)
```

###  2.3 transforms verification


```
def transforms_valdiation(dataset):
        print('type: {}\ndtype: {}\nnumber of images: {}\nimage_size: {}\n' .format(type(dataset.data),
                                                                            dataset.data.dtype,
                                                                            dataset.data.shape[0],
                                                                            dataset.data.shape[1]))

def data_valdiation(dataloader):
    print('type: {}\ndataset_size: {}\nnumber of batches: {}\nbatch_size: {}\n'.format(type(dataloader),
                                                                                len(dataloader.dataset),
                                                                                len(dataloader),
                                                                                dataloader.batch_size))

transforms_valdiation(trainset)
data_valdiation(trainloader)

```

**dataset.data** holds MNIST images. The data should be **Tensor** type with length of 60,000 were each element should be **uint8** with 28x28 pixels.
**dataloader** is a **DataLoader** type with size of 60,000. The batch_size is 64, thus dataloader length should be 60,000/64 = 937.5 (the last batch size is 32)

**output:**
```
type: <class 'torch.Tensor'>
dtype: torch.uint8
number of images: 60000
image_size: 28

type: <class 'torch.utils.data.dataloader.DataLoader'>
dataset_size: 60000
number of batches: 938
batch_size: 64
```

### 2.4 Exploring the data
displaying the images using [matplotlib.pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html). Since **tensor** type is iterable, i used **iter** and **next** to create an iterator object to read the first shuffled batch. Each batch from the training set consists of 64 images and 64 labels:

```
import matplotlib.pyplot as plt
import numpy as np

def data_integrity(data):
    data_iter = iter(data)
    images, labels = next(data_iter)
    count = [0] * 10
    digits = [i for i in range(10)]
    for label in labels:
        count[label] += 1
    figure = plt.figure(figsize=(8, 8))
    bars = plt.bar(digits, count, color='maroon', width=0.6)
    plt.xticks(np.arange(10), digits)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x()+ .25, yval + .2, yval)


def data_visualization(data):
    data_iter = iter(data)
    images, labels = next(data_iter)
    figure = plt.figure(figsize=(10,10))
    for index in range(1, data.batch_size+1):
        plt.subplot(8, 8, index)
        plt.axis('off')
        plt.imshow(images[index-1].numpy().squeeze(), cmap='gray_r')
        count[labels[index-1]] += 1

```











