classdef Loss < handle
    properties(Access = private)
        loss
        loss_derivative
    end
    
    methods(Access = public)
        function self = Loss(name)
            if name=="mse"
                self.loss = @(y_true, y_pred) 0.5*(y_true - y_pred).^2;
                self.loss_derivative = @(y_true, y_pred) (y_pred - y_true);
            end
        end
        
        function error = compute(self, y_true, y_pred)
            error = self.loss(y_true, y_pred);
        end
        
        function error = compute_derivative(self, y_true, y_pred)
            error = self.loss_derivative(y_true, y_pred);
        end
    end
end