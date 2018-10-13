classdef ActivationLayer < Layer
    properties(Access = private)
        activation
    end
    
    methods(Access = public)
        function self = ActivationLayer(input_size, activation)
            self.name = "activation_layer";
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
        
        function block = save(self)
            block{1} = self.input_size;
            block{2} = self.activation.get_name();
        end
    end
    
    methods(Static)
       function layer = load(block)
            input_size = block{1};
            activation = Activation(block{2});
            layer = ActivationLayer(input_size, activation);
        end 
    end
end