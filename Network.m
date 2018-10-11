classdef Network < handle
    properties(Access = private)
        layers = {};
    end
    
    methods(Access = public)
        function add_layer(self, layer)
            self.layers{size(self.layers,2) + 1} = layer;
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
        
        function [] = fit(self, input, output, epochs, learning_rate)
            layers_count = size(self.layers, 2);
            samples_count = size(input, 1);
            error = 0;
            for i=1:epochs
                for j=1:samples_count
                    pred = self.predict(input(j,:));
                    derror = pred - output(j,:);
                    error = error + mean((output(j,:) - pred).^2, 'all');
                    for k=layers_count:-1:1
                        derror = self.layers{1,k}.back_propagation(derror, learning_rate);
                    end
                end
                error = error / samples_count;
                fprintf('%d/%d   err=%f\n', i, epochs, error);
            end
        end
    end
end