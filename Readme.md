# AISC CPU Training Benchmarks

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

git clone https://github.com/rimman/AISC-CPU-Benchmarks.git
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

### 2. ImageNet

## Download the ImageNet dataset
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders.

1. Download the images from http://image-net.org/download-images

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

### 3. Training

There are 3 directories for training. The command below show how to run them.

**For the demo run the distributed data parallel training.**
**While the demo is running in another shell run `watch nvidia-smi` that shows the usage of the GPU.**

- Single GPU training

    ```bash
    cd single_gpu
    python3 train.py --batch_size 100
    # takes about 8 min
    ```

- Multi-GPU training (data parallel strategy)

    ```bash
    cd data_parallel
    python3 train.py --gpu_devices 0 1 --batch_size 100
    # takes about 4 min
    ```

- Multi-GPU training (distributed data parallel strategy)

    ```bash
    cd data_parallel
    python3 train.py --gpu_devices 0 1 --batch_size 100
    # takes about 4 min
    ```
