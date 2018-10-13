classdef (Abstract) Layer < handle
    properties(Access = protected)
        input_size
        output_size
        input_
        output_
    end

    properties(Access = public)
        name
    end
    
    methods(Abstract)
        output = forward_propagation(self, input);
        in_error = back_propagation(self, error, learning_rate);
        block = save(self);
    end
    
    methods(Abstract, Static)
        layer = load(block);
    end
    
    methods(Static)
        function layer = build_layer(name, block)
            if name=="fully_connected_layer"
                layer = FullyConnectedLayer.load(block);
            elseif name=="activation_layer"
                layer = ActivationLayer.load(block);
            end
        end
    end
    
    methods(Access = public)
        function in = get_input(self)
            in = self.input_;
        end
        
        function out = get_output(self)
            out = self.output_;
        end
        
        function is = get_input_size(self)
            is = self.input_size;
        end
        
        function os = get_output_size(self)
            os = self.output_size;
        end
    end
end