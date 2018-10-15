function [] = main()
    % add path to neural network
    addpath(genpath('../../sources'));
    
    % IO data
    input = [0 0 ; 0 1 ; 1 0 ; 1 1];
    output = [0 ; 1 ; 1 ; 0];
    
    % reshape data for neural network (sample dimension last)
    input = reshape(rot90(input), [1,2, 4]);
    output = reshape(rot90(output), [1,1, 4]);
    
    % create a 3-layer neural network
    net = Network();
    net.add_layer(FullyConnectedLayer([1 2], [1 3]));
    net.add_layer(ActivationLayer([1 3], Activation("exp")));
    net.add_layer(FullyConnectedLayer([1 3], [1 1]));
    net.add_layer(ActivationLayer([1 1], Activation("exp")));
    
    % set cost function and learning_rate
    net.build(Loss('mse'), 0.2)
    
    % train on 1000 iterations
    net.fit(input, output, 1000);
    
    % test network
    net.predict(input)
end