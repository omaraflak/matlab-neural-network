classdef ActivationLayer < Layer
    properties(Access = private)
        input_size
        activation
    end
    
    methods(Access = public)
        function self = ActivationLayer(input_size, activation)
            self.input_size = input_size;
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
        
        function is = get_input_size(self)
            is = self.input_size;
        end
        
        function os = get_output_size(self)
            os = self.input_size;
        end
    end
end