# Matlab Neural Network

This is a simple and modulat way of implementing a neural network with Matlab.

# Example

This is a regular neural network which implements the XOR problem.

```matlab
net = Network();
net.add_layer(FullyConnectedLayer(2, 3));
net.add_layer(ActivationLayer(3, @activation, @activation_prime));
net.add_layer(FullyConnectedLayer(3,1));
net.add_layer(ActivationLayer(1, @activation, @activation_prime));

input = [0 0 ; 0 1 ; 1 0 ; 1 1];
output = [0 ; 1 ; 1 ; 0];

net.fit(input, output, 1000, 0.2);
net.predict(input)
```
