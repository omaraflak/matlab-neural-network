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
        
        function [] = save(self, filename)
            layers_count = size(self.layers, 2);
            data = cell(1, layers_count);
            for k=1:layers_count
                data{1,k} = {self.layers{1,k}.get_name() self.layers{1,k}.save()};
            end
            save(filename, 'data');
        end
    end
    
    methods(Static)
        function net = load(filename)
            file = load(filename, 'data');
            data = file.data;
            layers_count = size(data, 2);
            net = Network();
            for i=1:layers_count
                layer_name = data{1,i}{1};
                block_data = data{1,i}{2};
                layer = Layer.build_layer(layer_name, block_data);
                net.add_layer(layer);
            end
        end
    end
end