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
