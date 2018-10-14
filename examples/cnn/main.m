function [] = main()
    addpath('../../sources');
    
    fprintf('loading training data...');
    [x_train, y_train, x_test, y_test] = read_training('training');
    fprintf('done.\n');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([32 32], [3 3], 1));
    net.add_layer(ActivationLayer([30 30 5], Activation('sigmoid')));
    net.add_layer(FlattenLayer([30 30 5]));
    net.add_layer(FullyConnectedLayer([1 4500], [1 10]));
    net.add_layer(ActivationLayer([1 10], Activation('sigmoid')));
    
    net.build(Loss('mse'), 0.7);
    
    fprintf('training...\n');
    net.fit(x_train, y_train, 30);
    
    fprintf('error on test data...\n');
    error = net.evaluate(x_test, y_test)
end