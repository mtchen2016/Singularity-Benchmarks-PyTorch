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
nvidia-smi

```

### 1. Sanity Check

Show that the PyTorch is installed successfully and can use see the GPUs.

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

### 2. Run Training

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
