### Distributed MNIST

See the [Readme](distributed_mnist/README.md) for description and execution.

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
