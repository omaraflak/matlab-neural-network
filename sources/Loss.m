classdef Loss < handle
    properties(Access = private)
        loss
        loss_derivative
    end
    
    methods(Access = public)
        function self = Loss(name)
            if name=="mse"
                self.loss = @(y_true, y_pred) mean((y_true - y_pred).^2, 'all');
                self.loss_derivative = @(y_true, y_pred) (2/numel(y_true))*(y_pred - y_true);
            elseif name=="cross-entropy"
                self.loss = @(y_true, y_pred) -sum(y_true*log10(y_pred), 'all');
                self.loss_derivative = @(y_true, y_pred) -y_true/(y_pred*log(10));
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