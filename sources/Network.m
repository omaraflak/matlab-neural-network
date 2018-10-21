classdef Network < handle
    properties(Access = private)
        layers = {};
        learning_rate
        loss
    end
    
    methods(Access = public)
        function self = Network(layers)
            self.layers = layers;
        end
        
        function [] = build(self, loss, learning_rate)
            self.loss = loss;
            self.learning_rate = learning_rate;
        end
        
        function output = predict(self, input)
            layers_count = length(self.layers);
            input_shape = self.layers{1}.get_input_size();
            output_shape = self.layers{layers_count}.get_output_size();
            
            samples_count = size(input, length(input_shape) + 1);
            otherdims = repmat({':'}, 1, length(input_shape));
            output = zeros([output_shape samples_count]);
                        
            for j=1:samples_count
                out = input(otherdims{:}, j);
                for i=1:layers_count
                    out = self.layers{i}.forward_propagation(out);
                end
                output(otherdims{:}, j) = out;
            end
        end
        
        function [] = fit(self, input, output, epochs)
            layers_count = length(self.layers);
            input_shape = self.layers{1}.get_input_size();
            samples_count = size(input, length(input_shape) + 1);
            otherdims = repmat({':'}, 1, length(input_shape));
            
            for i=1:epochs
                error = 0;
                for j=1:samples_count
                    % forward propagation
                    pred = input(otherdims{:}, j);
                    for k=1:layers_count
                        pred = self.layers{k}.forward_propagation(pred);
                    end
                
                    % backward propgation
                    derror = self.loss.compute_derivative(output(otherdims{:}, j), pred);
                    error = error + self.loss.compute(output(otherdims{:}, j), pred);
                    for k=layers_count:-1:1
                        derror = self.layers{k}.back_propagation(derror, self.learning_rate);
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
                data{1,k} = {self.layers{k}.get_name() self.layers{k}.save()};
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