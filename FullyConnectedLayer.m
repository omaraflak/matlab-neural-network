classdef FullyConnectedLayer < Layer
    properties(Access = private)
        weights
        bias
    end
    
    methods(Access = public)
        function self = FullyConnectedLayer(input_size, output_size)
            self.input_size = input_size;
            self.output_size = output_size;
            self.weights = rand(self.input_size, self.output_size) - 0.5;
            self.bias = rand(1, self.output_size) - 0.5;
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            self.output_ = input*self.weights + self.bias;
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, learning_rate)
            dWeights = self.input_' * error;
            dBias = error;
            self.weights = self.weights - learning_rate*dWeights;
            self.bias = self.bias - learning_rate*dBias;
            in_error = error * self.weights';
        end
    end
end