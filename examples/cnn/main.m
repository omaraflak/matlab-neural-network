function [] = main()
    addpath('../../sources');
    
    [x_train, y_train, x_test, y_test] = read_training('training');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([32 32 1], [3 3 1], 5));
    net.add_layer(ActivationLayer([30 30 5], Activation('sigmoid')));
    net.add_layer(FlattenLayer([30 30 5]));
    net.add_layer(FullyConnectedLayer(4500, 10));
    net.add_layer(ActivationLayer([1 10], Activation('sigmoid')));
    
    net.build(Loss('mse'), 0.7);
    
    net.fit(x_train, y_train, 30);
    error = net.evaluate(x_test, y_test)
end