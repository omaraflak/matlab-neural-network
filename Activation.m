classdef Activation < handle
    properties(Access = private)
        activation
        derivative
    end
    
    methods(Access = public)
        function self = Activation(name)
            if name=="tanh"
                self.activation = @(x) tanh(x);
                self.derivative = @(x) 1 - tanh(x)^2;
            elseif name=="sigmoid"
                self.activation = @(x) 1.0/(1.0+exp(-x));
                self.derivative = @(x) exp(-x)/((1+exp(-x))^2);
            elseif name=="relu"
                self.activation = @(x) max(0,x);
                self.derivative = @(x) max(0,x)/x;
            elseif name=="linear"
                self.activation = @(x) x;
                self.derivative = @(x) 1.0;
            end
        end
        
        function activation = get_activation(self)
            activation = self.activation;
        end
        
        function activation_prime = get_activation_prime(self)
            activation_prime = self.derivative;
        end
    end    
end