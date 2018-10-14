classdef FlattenLayer < Layer
    methods(Access = public)
        function self = FlattenLayer(input_size)
            self.input_size = input_size;
            self.output_size = [1 prod(input_size)];
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            self.output_ = reshape(input, 1, []);
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, ~)
            in_error = reshape(error, self.input_size);
        end
        
        function block = save(self)
            block{1} = self.input_size;
        end
    end
    
    methods(Static)
        function layer = load(block)
            input_size = block{1};
            layer = FlattenLayer(input_size);
        end
        
        function name = get_name()
            name = "flatten_layer";
        end
    end
end