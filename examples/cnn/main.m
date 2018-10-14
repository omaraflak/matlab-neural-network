function [] = main()
    addpath('../../sources');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([10 10], [5 5], 2));
    net.add_layer(ActivationLayer([6 6 2], Activation('sigmoid')));
    net.add_layer(MaxPoolLayer([6 6 2], [3 3]));
    net.add_layer(ConvolutionalLayer([4 4 2], [3 3], 1));
    net.add_layer(ActivationLayer([2 2 2], Activation('sigmoid')));
    net.add_layer(FlattenLayer([2 2 2]));
    
    net.build(Loss('mse'), 0.1);
       
    input = rand(10,10);
    output = rand(1,8);
    
    net.fit(input, output, 100);
    error = net.evaluate(input, output)
end