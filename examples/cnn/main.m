function [] = main()
    addpath('../../sources');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([5 5], [3 3], 1));
    
    % net.build(Loss('mse'), 0.1);
    
    input = rand(5);
    
    net.predict(input);
end