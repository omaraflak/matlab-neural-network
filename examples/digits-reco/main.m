function [] = main()
    % add path to neural network
    addpath('../../sources');
    
    % load data from training file
    fprintf("loading data...");
    [x_train, y_train, x_test, y_test] = read_training('training');
    fprintf("done.\n");
    
    % create a 3-layer neural network
    net = Network();
    net.add_layer(FullyConnectedLayer(32*32, 15));
    net.add_layer(ActivationLayer([1 15], Activation('sigmoid')));
    net.add_layer(FullyConnectedLayer(15, 10));
    net.add_layer(ActivationLayer([1 10], Activation('sigmoid')));
    
    % set cost function and learning rate
    net.build(Loss('mse'), 0.2);
    
    % train on 30 iterations
    fprintf("training...\n");
    net.fit(x_train, y_train, 30);
    
    % calculate error
    error = net.evaluate(x_test, y_test);
    fprintf("error on test data = %f\n", error);    
end