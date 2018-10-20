classdef (Abstract) Layer < handle
    properties(Access = protected)
        input_size
        output_size
        input_
        output_
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
            if name==FullyConnectedLayer.get_name()
                layer = FullyConnectedLayer.load(block);
            elseif name==ActivationLayer.get_name()
                layer = ActivationLayer.load(block);
            elseif name==ConvolutionalLayer.get_name()
                layer = ConvolutionalLayer.load(block);
            elseif name==MaxPoolLayer.get_name()
                layer = MaxPoolLayer.load(block);
            elseif name==FlattenLayer.get_name()
                layer = FlattenLayer.load(block);
            elseif name==DropoutLayer.get_name()
                layer = DropoutLayer.load(block);
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
    
    methods(Access = public, Static)
        name = get_name();
    end
end