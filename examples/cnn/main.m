function [] = main()
    addpath('../../sources');
    
    net = Network();
    net.add_layer(ConvolutionalLayer([10 10], [3 3], 1));
    
    net.build(Loss('mse'), 0.1);
       
    input = rand(10,10);
    output = rand(8,8);
    
    net.fit(input, output, 100);
    
    net.predict(input)
end