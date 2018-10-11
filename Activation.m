classdef Activation < handle
    properties(Access = private)
        activation
        derivative
    end
    
    methods(Access = public)
        function self = Activation(name)
            if name=="sigmoid"
                self.activation = @(x) 1.0/(1.0+exp(-x));
                self.derivative = @(x) exp(-x)/((1+exp(-x))^2);
            elseif name=="tanh"
                self.activation = @(x) tanh(x);
                self.derivative = @(x) 1 - tanh(x)^2;
            elseif name=="relu"
                self.activation = @(x) max(0,x);
                self.derivative = @(x) max(0,x)/x;
            elseif name=="linear"
                self.activation = @(x) x;
                self.derivative = @(x) 1.0;
            elseif name=="leaky_relu"
                self.activation = @self.leaky_relu;
                self.derivative = @self.leaky_relu_der;
            end
        end
        
        function activation = get_activation(self)
            activation = self.activation;
        end
        
        function activation_prime = get_activation_prime(self)
            activation_prime = self.derivative;
        end
        
        % leaky relu
        function y = leaky_relu(self, x)
            if x<0
                y = 0.01*x;
            else
                y = 0;
            end
        end
        
        % leaky_relu derivative
        function y = leaky_relu_der(self, x)
            if x<0
                y = 0.01;
            else
                y = 0;
            end
        end
    end    
end