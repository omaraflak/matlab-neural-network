# Matlab Neural Network

A simple and modular way of implementing a neural network with Matlab.

# Example
Simple neural network applied for the XOR problem. See [main.m](https://github.com/OmarAflak/matlab-neural-network/blob/master/examples/xor/main.m) file.

```matlab
input = [0 0 ; 0 1 ; 1 0 ; 1 1];
output = [0 ; 1 ; 1 ; 0];

net = Network();
net.add_layer(FullyConnectedLayer(2, 3));
net.add_layer(ActivationLayer(3, Activation("tanh")));
net.add_layer(FullyConnectedLayer(3,1));
net.add_layer(ActivationLayer(1, Activation("tanh")));

net.build(Loss('mse'), 0.2);
net.fit(input, output, 1000);
net.predict(input)
```
