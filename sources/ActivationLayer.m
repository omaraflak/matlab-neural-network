classdef ActivationLayer < Layer
    properties(Access = private)
        activation
    end
    
    methods(Access = public)
        function self = ActivationLayer(input_size, activation)
            self.input_size = input_size;
            self.output_size = input_size;
            self.activation = activation;
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            self.output_ = arrayfun(self.activation.get_activation(), input);
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, ~)
            in_error = times(arrayfun(self.activation.get_activation_prime(), self.input_), error);
        end
    end
end