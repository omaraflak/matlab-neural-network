function [] = main()
    net = Network();
    net.add_layer(FullyConnectedLayer(2, 3));
    net.add_layer(ActivationLayer(3, @sigmoid, @sigmoid_prime));
    net.add_layer(FullyConnectedLayer(3,1));
    net.add_layer(ActivationLayer(1, @sigmoid, @sigmoid_prime));
    
    input = [0 0 ; 0 1 ; 1 0 ; 1 1];
    output = [0 ; 1 ; 1 ; 0];
    
    net.fit(input, output, 1000, 0.1);
    net.predict(input)
end

function y = sigmoid(x)
    % y = 1/(1+exp(-x));
    y = tanh(x);
end

function y = sigmoid_prime(x)
    % y = exp(-x)/((1+exp(-x))^2);
    y = 1 - tanh(x)^2;
end