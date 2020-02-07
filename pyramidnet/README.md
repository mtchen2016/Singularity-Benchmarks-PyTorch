### PyramidNet

See the [Readme](pyramid_net/README.md) for description and execution.

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
