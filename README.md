# Matlab Neural Network

A simple and modular way of implementing a neural network with Matlab.

# Layers

* FullyConnectedLayer(input_shape, output_shape)
* ConvolutionalLayer(input_shape, kernel_shape, layer_depth)
* MaxPoolLayer(input_shape, kernel_shape)
* FlattenLayer(input_shape)
* ActivationLayer(input_shape, activation)

# Activations

* Sigmoid ('sigmoid')
* Hyperbolic Tangant ('tanh')
* Rectified Linear Unit ('relu')
* Leaky Rectified Linear Unit ('leaky_relu')
* Linear ('linear')
* Exponential ('exp')
* Softplus ('softplus')
* Softsign ('softsign')

# Losses

* Mean Squared Error ('mse')
* Mean Squared Logarithmic Error ('msle')
* Mean Absolute Error ('mae')
* Negative Logarithmic Likelihood ('neg_log_likelihood')
* Cross Entropy ('cross_entropy')

# Example
Simple neural network applied to the XOR problem. See all examples **[here](https://github.com/OmarAflak/matlab-neural-network/blob/master/examples)**.

```matlab
    % IO data
    input = [0 0 ; 0 1 ; 1 0 ; 1 1];
    output = [0 ; 1 ; 1 ; 0];
    
    % reshape data for neural network (sample dimension last)
    input = reshape(rot90(input), [1,2, 4]);
    output = reshape(rot90(output), [1,1, 4]);
    
    % create a 3-layer neural network
    net = Network();
    net.add_layer(FullyConnectedLayer([1 2], [1 3]));
    net.add_layer(ActivationLayer([1 3], Activation("tanh")));
    net.add_layer(FullyConnectedLayer([1 3], [1 1]));
    net.add_layer(ActivationLayer([1 1], Activation("tanh")));
    
    % set cost function and learning_rate
    net.build(Loss('mse'), 0.2)
    
    % train on 1000 iterations
    net.fit(input, output, 1000);
    
    % test network
    net.predict(input)
```

# Save / Load Network

```matlab
% save network
net.save('network.mat');

% load network
net = Network.load('network.mat');
```
