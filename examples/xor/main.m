function [] = main()
    % add path to neural network
    addpath('../../sources');
    
    % create a 3-layer neural network
    net = Network();
    net.add_layer(FullyConnectedLayer(2, 3));
    net.add_layer(ActivationLayer(3, Activation("tanh")));
    net.add_layer(FullyConnectedLayer(3,1));
    net.add_layer(ActivationLayer(1, Activation("tanh")));
    
    % IO data
    input = [0 0 ; 0 1 ; 1 0 ; 1 1];
    output = [0 ; 1 ; 1 ; 0];
    
    % start training on 1000 iterations, with a learning rate of 0.2
    net.fit(input, output, Loss("mse"), 1000, 0.2);
    
    % test network
    net.predict(input)
end