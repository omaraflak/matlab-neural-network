function [] = test_network()
    % add path to neural network
    addpath('../../sources');
    
    % load data from training file
    fprintf("loading data...");
    [x_train, y_train, x_test, y_test] = read_training('training');
    fprintf("done.\n");
    
    % create a 3-layer neural network
    net = Network();
    net.add_layer(FullyConnectedLayer(32*32, 15));
    net.add_layer(ActivationLayer(15, Activation('sigmoid')));
    net.add_layer(FullyConnectedLayer(15, 10));
    net.add_layer(ActivationLayer(10, Activation('sigmoid')));
    
    % train on 30 iterations, with a learning rate of 0.2
    fprintf("training...\n");
    loss = Loss('mse');
    net.fit(x_train, y_train, loss, 30, 0.2);
    
    % test network on new data
    output = net.predict(x_test);
    output = arrayfun(@step_function, output);
    
    % calculate error
    error = loss.compute(y_test, output);
    fprintf("total error = %f\n", error);    
end

function output = step_function(input)
   if input > 0.9
       output = 1.0;
   elseif input < 0.1
       output = 0.0;
   else
       output = input;
   end
end