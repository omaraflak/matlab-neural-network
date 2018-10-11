function [] = main()
    net = Network();
    net.add_layer(FullyConnectedLayer(2, 3));
    net.add_layer(ActivationLayer(3, @activation, @activation_prime));
    net.add_layer(FullyConnectedLayer(3,1));
    net.add_layer(ActivationLayer(1, @activation, @activation_prime));
    
    input = [0 0 ; 0 1 ; 1 0 ; 1 1];
    output = [0 ; 1 ; 1 ; 0];
    
    net.fit(input, output, 1000, 0.2);
    net.predict(input)
end

function y = activation(x)
    y = tanh(x);
end

function y = activation_prime(x)
    y = 1 - tanh(x)^2;
end