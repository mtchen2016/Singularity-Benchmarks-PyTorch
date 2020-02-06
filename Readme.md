# AISC Training Benchmarks

This repository contains the scripts and setup instructions for training models using CPU only.

## Setup

### 1. Install PyTorch and other utilities

Install Git, VIM, Python, PyTorch and other required packages with the following commands.

```bash
# run the following two commands for LCOW only
apt-get update
apt-get install -y sudo

# run the following commands for both Azure VM and LCOW
sudo apt-get install -y git
sudo apt-get install -y vim
sudo apt-get -y install python3
sudo apt-get -y install python3-pip
python3 -m pip install torch
python3 -m pip install torchvision
python3 -m pip install tensorboardX

```

### 2. Clone the demo repository

```bash

git clone https://github.com/rimman/AISC-Benchmarks.git
cd AISC-CPU-Benchmarks
```

## Models

### 1. MNIST

Show the list of possible arguments through `--help` command.

```bash
python3 mnist.py --help

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 64)
  --test-batch-size N  input batch size for testing (default: 1000)
  --epochs N           number of epochs to train (default: 14)
  --lr LR              learning rate (default: 1.0)
  --gamma M            Learning rate step gamma (default: 0.7)
  --no-cuda            disables CUDA training
  --seed S             random seed (default: 1)
  --log-interval N     how many batches to wait before logging training status
  --save-model         For Saving the current Model
  --num_cpus N         number of CPU vCores to train with (default: use all
                       available)
```

Run the training as follows:

```bash
python mnist.py --num_cpus 12 --epochs 10 
```

### 2. Fashion MNIST

Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Fashion-MNIST can serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

Run the training as follows:

```bash
python fashionmnist.py 
```

### 3. Image Classifier

Here, we will use the CIFAR10 dataset. It has the classes: ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’. The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size.

Training an image classifier code will do the following steps in order:
- Load and normalizing the CIFAR10 training and test datasets using torchvision
- Define a Convolutional Neural Network
- Define a loss function
- Train the network on the training data
- Test the network on the test data

### 4. ImageNet

#### Download the ImageNet dataset
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images or ask Rimma (she has a local copy of data). You will need **ILSVRC2012_img_train.tar** and **ILSVRC2012_img_val.tar**.

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```

#### Training

To train a model, run `main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python main.py -a resnet18 [imagenet-folder with train and val folders]
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

```bash
python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]
```

#### Multi-processing Distributed Data Parallel Training

You should always use the NCCL backend for multi-processing distributed training since it currently provides the best distributed training performance.

##### Single node, multiple GPUs:

```bash
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 [imagenet-folder with train and val folders]
```

##### Multiple nodes:

Node 0:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0 [imagenet-folder with train and val folders]
```

Node 1:
```bash
python main.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1 [imagenet-folder with train and val folders]
```

#### Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | resnet | resnet101 |
                        resnet152 | resnet18 | resnet34 | resnet50 | vgg |
                        vgg11 | vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn
                        | vgg19 | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
```


### 5. Toy

This is a very basic example to validate distributed data parallel training. To run the code, use the following commands:

**Worker - Rank 0**
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2
```

**Worker - Rank 1**
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2
```


### 6. Distributed MNIST

This is a sample that trains MNIST in a distributed manner. To run the script use the following commands:

**Worker - Rank 0**
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2
```

**Worker - Rank 1**
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2
```

#### Use specific root directory for running example on single machine.

**Worker - Rank 0**
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2 --root data0
```

**Worker - Rank 1**
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2 --root data1
```

### 7. PyramidNet (Multi-GPU training)
This is multi GPU training code for PyramidNet with PyTorch. You will train PyramidNet for CIFAR10 classification task. This code is for comparing several ways of multi-GPU training. There are 3 directories for training. The command below show how to run them.
While the code is running in another shell run `watch nvidia-smi` that shows the usage of the GPU.

- **Single GPU training**

    ```bash
    cd single_gpu
    python3 train.py --batch_size 100
    # takes about 8 min
    ```

- **Multi-GPU training (data parallel strategy)**

    ```bash
    cd data_parallel
    python3 train.py --gpu_devices 0 1 --batch_size 100
    # takes about 4 min
    ```

- **Multi-GPU training (distributed data parallel strategy)**

    ```bash
    cd data_parallel
    python3 train.py --gpu_devices 0 1 --batch_size 100
    # takes about 4 min
    ```
