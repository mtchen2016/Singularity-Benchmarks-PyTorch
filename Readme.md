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

See the [Readme](mnist/README.md) for description and execution.

### 2. Fashion MNIST

See the [Readme](fashion_mnist/README.md) for description and execution.

### 3. Image Classifier

See the [Readme](classifier/README.md) for description and execution.

### 4. ImageNet



### 5. Toy

This is a very basic example to validate distributed data parallel training. To run the code, use the following commands:

##### Worker - Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2
```

##### Worker - Rank 1
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2
```


### 6. Distributed MNIST

This is a sample that trains MNIST in a distributed manner. To run the script use the following commands:

##### Worker - Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2
```

##### Worker - Rank 1
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2
```

#### Use specific root directory for running example on single machine.

##### Worker - Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2 --root data0
```

##### Worker - Rank 1
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2 --root data1
```

### 7. PyramidNet
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

### 8. Word-level language modeling RNN

This script trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash 
python main.py --cuda --epochs 6           # Train a LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --tied    # Train a tied LSTM on Wikitext-2 with CUDA
python main.py --cuda --epochs 6 --model Transformer --lr 5   
                                           # Train a Transformer model on Wikitext-2 with CUDA
python main.py --cuda --tied               # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs
python generate.py                         # Generate samples from the trained LSTM model.
python generate.py --cuda --model Transformer
                                           # Generate samples from the trained Transformer model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help                       show this help message and exit
  --data DATA                      location of the data corpus
  --model MODEL                    type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE                  size of word embeddings
  --nhid NHID                      number of hidden units per layer
  --nlayers NLAYERS                number of layers
  --lr LR                          initial learning rate
  --clip CLIP                      gradient clipping
  --epochs EPOCHS                  upper epoch limit
  --batch_size N                   batch size
  --bptt BPTT                      sequence length
  --dropout DROPOUT                dropout applied to layers (0 = no dropout)
  --decay DECAY                    learning rate decay per epoch
  --tied                           tie the word embedding and softmax weights
  --seed SEED                      random seed
  --cuda                           use CUDA
  --log-interval N                 report interval
  --save SAVE                      path to save the final model
  --transformer_head N             the number of heads in the encoder/decoder of the transformer model
  --transformer_encoder_layers N   the number of layers in the encoder of the transformer model
  --transformer_decoder_layers N   the number of layers in the decoder of the transformer model
  --transformer_d_ff N             the number of nodes on the hidden layer in feed forward nn
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40           
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied    
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40        
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied 
```
