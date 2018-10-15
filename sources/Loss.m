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
            elseif name=="msle"
                self.loss = @(y_true, y_pred) mean((log10(y_true+1)-log10(y_pred+1)).^2, 'all');
                self.loss_derivative = @(y_true, y_pred) (-2/numel(y_true))*(log10(y_true+1)-log10(y_pred+1))/((log(10)*(y_pred+1)));
            elseif name=="mae"
                self.loss = @(y_true, y_pred) mean(abs(y_true - y_pred), 'all');
                self.loss_derivative = @(y_true, y_pred) (y_pred - y_true)./(numel(y_true)*abs(y_true - y_pred));
            elseif name=="neg_log_likelihood"
                self.loss = @(y_true, y_pred) -mean(log10(y_pred), 'all');
                self.loss_derivative = @(y_true, y_pred) 1./(log(10)*y_pred);
            elseif name=="cross_entropy"
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