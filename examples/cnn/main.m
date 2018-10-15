function [] = main()
    addpath(genpath('../../sources'));
    
    fprintf('loading training data...');
    [x_train, y_train, x_test, y_test] = read_training('training');
    fprintf('done.\n');
        
    net = Network();
    net.add_layer(ConvolutionalLayer([32 32], [5 5], 3));
    net.add_layer(ActivationLayer([28 28 3], Activation('sigmoid')));
    net.add_layer(MaxPoolLayer([28 28 3], [9 9]));
    net.add_layer(FlattenLayer([20 20 3]));
    net.add_layer(FullyConnectedLayer([1 1200], [1 10]));
    net.add_layer(ActivationLayer([1 10], Activation('sigmoid')));
    
    net.build(Loss('mse'), 0.7);
    
    fprintf('training...\n');
    net.fit(x_train, y_train, 30);
    
    fprintf('error on test data...\n');
    error = net.evaluate(x_test, y_test)
end