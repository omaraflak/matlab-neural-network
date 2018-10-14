function [] = main()
    addpath('../../sources');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([10 10], [5 5], 1));
    net.add_layer(ActivationLayer([6 6], Activation('sigmoid')));
    net.add_layer(ConvolutionalLayer([6 6], [3 3], 1));
    net.add_layer(ActivationLayer([4 4], Activation('sigmoid')));
    
    net.build(Loss('mse'), 0.1);
       
    input = rand(10,10);
    output = rand(4,4);
    
    net.fit(input, output, 100);
    net.predict(input)
    output
    error = net.evaluate(input, output)
end