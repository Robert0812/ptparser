# ptparser
A parser for profiling model structures and parameters only with the caffe model prototxt 

# Example

    python ptparser.py -p lenet_train_test.prototxt -w 28 -g 28
    
```
CONV(conv1): [25x25x20]
params: (5x5x3)x20 = 1500
POOL(pool1): [12x12x20]
params: 0
CONV(conv2): [9x9x50]
params: (5x5x20)x50 = 25000
POOL(pool2): [4x4x50]
params: 0
FC(ip1): [1x1x500]
params: 800x500 = 400000
FC(ip2): [1x1x10]
params: 500x10 = 5000
TOTAL params: 1.6 MB parameters
```
    
