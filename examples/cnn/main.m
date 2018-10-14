function [] = main()
    addpath('../../sources');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([10 10 1], [5 5 1], 2));
    net.add_layer(ActivationLayer([6 6 2], Activation('sigmoid')));
    net.add_layer(ConvolutionalLayer([6 6 2], [3 3 2], 1));
    net.add_layer(ActivationLayer([4 4 2], Activation('sigmoid')));
    
    net.build(Loss('mse'), 0.1);
       
    input = rand(10,10);
    output = rand(4,4,2);
    
    net.fit(input, output, 100);
    error = net.evaluate(input, output)
end