function [] = main()
    % add path to neural network
    addpath('../../sources');
    
    % IO data
    input = [0 0 ; 0 1 ; 1 0 ; 1 1];
    output = [0 ; 1 ; 1 ; 0];
    
    % create a 3-layer neural network
    net = Network();
    net.add_layer(FullyConnectedLayer(2, 3));
    net.add_layer(ActivationLayer(3, Activation("tanh")));
    net.add_layer(FullyConnectedLayer(3,1));
    net.add_layer(ActivationLayer(1, Activation("tanh")));
    
    % set cost function and learning_rate
    net.build(Loss('mse'), 0.2)
    
    % train on 1000 iterations
    net.fit(input, output, 1000);
    
    % test network
    net.predict(input)
end