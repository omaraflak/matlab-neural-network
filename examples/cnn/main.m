function [] = main()
    addpath('../../sources');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([10 10 1], [5 5 1], 1));
    net.add_layer(ActivationLayer([6 6 1], Activation('sigmoid')));
    net.add_layer(MaxPoolLayer([6 6 1], [3 3 1]));
    net.add_layer(ConvolutionalLayer([4 4 1], [3 3 1], 1));
    net.add_layer(ActivationLayer([2 2 1], Activation('sigmoid')));
    net.add_layer(FlattenLayer([2 2 1]));
    
    net.build(Loss('mse'), 0.2);
       
    input = rand(10,10);
    output = rand(1,4);
    
    net.fit(input, output, 100);
    error = net.evaluate(input, output)
end