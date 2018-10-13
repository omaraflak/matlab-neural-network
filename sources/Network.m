classdef Network < handle
    properties(Access = private)
        layers = {};
        learning_rate
        loss
    end
    
    methods(Access = public)
        function add_layer(self, layer)
            self.layers{size(self.layers,2) + 1} = layer;
        end
        
        function [] = build(self, loss, learning_rate)
            self.loss = loss;
            self.learning_rate = learning_rate;
        end
        
        function output = predict(self, input)            
            layers_count = size(self.layers, 2);
            samples_count= size(input, 1);
            output = zeros(samples_count, self.layers{layers_count}.get_output_size());
            for j=1:samples_count
                out = input(j,:);
                for i=1:layers_count
                    out = self.layers{1,i}.forward_propagation(out);
                end
                output(j,:) = out;
            end
        end
        
        function [] = fit(self, input, output, epochs)
            layers_count = size(self.layers, 2);
            samples_count = size(input, 1);
            error = 0;
            for i=1:epochs
                for j=1:samples_count
                    pred = self.predict(input(j,:));
                    derror = self.loss.compute_derivative(output(j,:), pred);
                    error = error + self.loss.compute(output(j,:), pred);
                    for k=layers_count:-1:1
                        derror = self.layers{1,k}.back_propagation(derror, self.learning_rate);
                    end
                end
                error = error / samples_count;
                fprintf('%d/%d   err=%f\n', i, epochs, error);
            end
        end
        
        function [error] = evaluate(self, x_test, y_test)
            y_pred = self.predict(x_test);
            error = self.loss.compute(y_test, y_pred);
        end
    end
end