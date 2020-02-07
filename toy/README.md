### Toy

See the [Readme](toy/README.md) for description and execution.

This is a very basic example to validate distributed data parallel training. To run the code, use the following commands:

##### Worker - Rank 0
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 0 --world-size 2
```

##### Worker - Rank 1
```
$ python3 main.py --init-method tcp://127.0.0.1:23456 --rank 1 --world-size 2
```
