# MNIST_Classifier
Digit recognition with multilayer perceptron classifier using PyTorch.  

## Contents
1. Objective
2. Dataset
3. Neural Network


## 1. Objective
This project aims to:
1. Develop a **handwritten digit classifier** on [MNIST dataset](http://yann.lecun.com/exdb/mnist/) 
2. Step-by-step guide.

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
displaying the images using [matplotlib.pyplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html). Since **tensor** type is iterable, i used **iter** and **next** to create an iterator object to read the first shuffled batch. Each batch from the training set consists of 64 images and 64 labels. To explor data integrity, i also verified the digits distribution in a random batch:   

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


data_visualization(trainloader)
data_integrity(trainloader)

```
the output will be 64 random images from the 1st batch in 8 rows and columns and a bar plot (x-axis 0-9 digits and y-axis is the count of each digits in the current batch):

![Figure_1](https://user-images.githubusercontent.com/57630290/188660788-51cf2701-950d-453c-aa8d-9f361e617e23.png)

![Figure_2](https://user-images.githubusercontent.com/57630290/188660446-bd089961-c5f1-4ba6-b061-5b5719481925.png)

## 3. Neural Network
to build a NN, i need to address some basics concepts:
1. Layered architecture
2. Activation functions
3. Loss function
4. Optimizer 

### 3.1 Layered architecture
I created a Two hidden layers architecture: one input layer, two hidden layers and one ouput. The first layer's neurons are reading the image, passing the pixels to the first hidden layer after multiplication of the pixels with the weights. The hidden layers will add bias to each entery, sum all the results and push it to the activaqtion functions. The output layer is passing the 10 values (number of classifications = 0-9 digits) to the Loss function. Thus, the size of the output layer is 10.           
I build the NN with [Torch.NN](https://pytorch.org/docs/stable/nn.html), with sequential container. The forward() method of Sequential accepts any input and forwards it to the first module it contains. It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.   


1. Input layer - the input layer will read the full image (consists 784 pixels) and each neuron is connected to all the neurons in the first hidden layer.  
2. Hidden layers - Two hidden layers with 128 entries and 64 each. The Wx multiplication is a matrix-vector calculation. The W-matrix have 128 weights for each pixel. Thus, W matrix size = (128,784) with 128 neurons and 784 enteries. The next hidden layer shopuld have a W matrix of 64 rows and 128 columns. 
3. Output layer - after the hidden layers, the NN have 64 values that corrisponding to 0-9 digits. The output layer will read the vector to convert all the 64 values into 10 values.        

The total network will look like this:
![1_HWhBextdDSkxYvz0kEMTVg](https://user-images.githubusercontent.com/57630290/188683133-892eebbb-4bd4-40de-8dd4-83a60f42443f.png)

The [torch.nn](https://pytorch.org/docs/stable/nn.html) contains the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) module which applies a linear transformation to the incoming data. After the linear operation, we can push the sum to the activation function.
### 3.2 Activation functions
**ReLU function** - ReLU(x) = max(0,x).
```
from torch import nn

def neural_network_validation(model):
    print('total modules: {}\n'.format(len(model._modules)))
    for key in model._modules.keys():
        if model._modules[key]._get_name() is not 'ReLU':
            dim = model._modules[key].weight.shape
            trans_num = 1 + (3-int(key)) % 3
            print("linear transformation step: {}\ninput: {}\noutput: {}\nW-matrix: {}\n".format(trans_num, dim[1], dim[0], dim))

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                    nn.ReLU(),
                    nn.Linear(hidden_sizes[1], output_size))
neural_network_validation(model)    

```
**output:**
```
total modules: 5
linear transformation step: 1
input: 784
output: 128
W-matrix: torch.Size([128, 784])

linear transformation step: 2
input: 128
output: 64
W-matrix: torch.Size([64, 128])

linear transformation step: 3
input: 64
output: 10
W-matrix: torch.Size([10, 64])
```
the output as expected - first neuron layer holds 1 full image of 784 pixels and the first hidden layer outputs 128 values. Each value is a sum of all the weights multiplied by the corrisponding pixel. the next hidden layer is reading the 128 values in each neuron after multiplication with the new weights. The output is 64 values as an input to the last linear operation, and the output is 10 values corrisponding to 10 digits. 

### 3.2 Loss function 
**cross entropy** - The cross entropy is the negative log of the **Sofmax function**.

```
def loss_function_validation(input, loss):
    print('number of images: {}\nnumber of classes: {}\ntotal loss: {}'.format(len(input), len(input[0]), loss))

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)
criterion = nn.CrossEntropyLoss()
output = model(images)
loss = criterion(output, labels)

loss_function_validation(output, loss):

```
i used the .view() operation to flatten out each image before sending it inside the neural network.
**output:**

```
number of images: 64
number of classes: 10
total loss: 2.309812545776367
```

### 3.3 Optimizer
[optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) implements stochastic gradient descent. The implementation of SGD with Momentum:
```
v(t+1) = m * v(t) + g(t+1)
P(t+1) = P(t) - lr * v(t+1) 
```
- **model.parameters()** (=P(t)) - every iteration we update the parameters.    
- **lr** - factor for the gradient's influence on the parameter's change.  
- **momentum** - (=m) factor for the influence of velocity.    
each image from the 64 images in the output, is holding 10 scores between 0-9. The loss function is reading each image seperatly.   

```
from torch import nn, optim

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
```
i used the optim.zero_grad() to set all gradients to zero before running a new batch and optim.step() after computing the Loss.
the training code: 

```
    time0 = time()
    epochs = 15
    epochs_dic = {}
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            epochs_dic[e]= running_loss / len(trainloader)
    print("\nTraining Time (in minutes) =", (time() - time0) / 60)
    plot_loss(epochs_dic)

```
the forward function includes flattening the images vector, set gradients to zero and model prediction.
the backward function includes the loss computation, loss.bacwards() and the optimizer step.   

the output: 



