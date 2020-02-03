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

Show the GPU information through `nvidia-smi` command.

```bash
nvidia-smi

```

### 1. Sanity Check

Show that the PyTorch is installed successfully and can use see the GPUs.

```bash
python3 verify-setup.py

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
