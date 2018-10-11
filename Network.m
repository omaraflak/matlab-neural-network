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
            input_count = size(input, 1);
            output = zeros(input_count, self.layers{layers_count}.get_output_size());
            for j=1:input_count
                out = input(j,:);
                for i=1:layers_count
                    out = self.layers{1,i}.forward_propagation(out);
                end
                output(j,:) = out;
            end
        end
        
        function [] = fit(self, input, output, epochs, learning_rate)
            layers_count = size(self.layers, 2);
            input_count = size(input, 1);
            err = 0;
            for i=1:epochs
                pred = self.predict(input);
                for j=1:input_count
                    error = (output(j,:) - pred(j,:)).^2;
                    err = err + mean(error, 'all');
                    for k=layers_count:-1:1
                        error = self.layers{1,k}.back_propagation(error, learning_rate);
                    end
                end
                err = err / input_count;
                fprintf('%d/%d   err=%f\n',i,epochs,err);
            end
        end
    end
end