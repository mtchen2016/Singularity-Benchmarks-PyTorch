# AISC Training Benchmarks

This repository contains the scripts and setup instructions for training models.

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
cd AISC-Benchmarks
```

## PyTorch Models

### 1. MNIST

See the [Readme](mnist/README.md) for description and execution.

### 2. Fashion MNIST

See the [Readme](fashion_mnist/README.md) for description and execution.

### 3. Image Classifier

See the [Readme](classifier/README.md) for description and execution.

### 4. ImageNet 
**Models:** alexnet, densenet121, densenet161, densenet169, densenet201, resnet101, resnet152, resnet18, resnet34, resnet50, squeezenet1_0, squeezenet1_1, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19 

See the [Readme](imagenet/README.md) for description and execution.

### 5. Toy

See the [Readme](toy/README.md) for description and execution.

### 6. Distributed MNIST

See the [Readme](distributed_mnist/README.md) for description and execution.

### 7. PyramidNet

See the [Readme](pyramid_net/README.md) for description and execution.

### 8. Word-level language RNN

See the [Readme](word_language_model/README.md) for description and execution.

## MLPerf Benchmark

This directory contains reference implementations for the MLPerf benchmark.
There are implementations for each of the 7 benchmarks in the MLPerf suite.

- image_classification - Resnet-50 v1 applied to Imagenet.
- object_detection - Mask R-CNN applied to COCO.
- single_stage_detector - SSD applied to COCO 2017.
- speech_recognition - DeepSpeech2 applied to Librispeech.
- translation - Transformer applied to WMT English-German.
- recommendation - Neural Collaborative Filtering applied to MovieLens 20 Million (ml-20m).
- sentiment_analysis - Seq-CNN applied to IMDB dataset.
- reinforcement - Mini-go applied to predicting pro game moves.
- Each reference implementation provides the following:

### 1. image_classification

See the [Readme](mlpeft/image_classification/README.md) for description and execution.

### 2. object_detection

See the [Readme](mlpeft/object_detection/README.md) for description and execution.

### 3. single_stage_detector

See the [Readme](mlpeft/single_stage_detector/README.md) for description and execution.

### 4. speech_recognition

See the [Readme](mlpeft/speech_recognition/README.md) for description and execution.

### 5. translation

See the [Readme](mlpeft/translation/README.md) for description and execution.
