classdef FullyConnectedLayer < Layer
    properties(Access = private)
        weights
        bias
    end
    
    methods(Access = public)
        function self = FullyConnectedLayer(input_size, output_size)
            self.name = "fully_connected_layer";
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
        
        function block = save(self)
            block{1} = self.input_size;
            block{2} = self.output_size;
            block{3} = self.weights;
            block{4} = self.bias;
        end
    end
    
    methods(Static)
        function layer = load(block)
            input_size = block{1};
            output_size = block{2};
            layer = FullyConnectedLayer(input_size, output_size);
            layer.weights = block{3};
            layer.bias = block{4};
        end
    end
end