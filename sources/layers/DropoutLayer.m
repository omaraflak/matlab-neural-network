classdef DropoutLayer < Layer
    properties(Access = private)
        drop_rate
        index
    end
    
    methods(Access = public)
        function self = DropoutLayer(input_size, drop_rate)
            self.input_size = input_size;
            self.output_size = input_size;
            self.drop_rate = drop_rate;
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            num = numel(input);
            self.index = randperm(num, int8(self.drop_rate*num));
            input(self.index) = 0;
            self.output_ = input;
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, ~)
            error(self.index) = 0;
            in_error = error;
        end
        
        function block = save(self)
            block{1} = self.input_size;
            block{2} = self.drop_rate;
        end
    end
        
    methods(Static)
       function layer = load(block)
            input_size = block{1};
            drop_rate = block{2};
            layer = DropoutLayer(input_size, drop_rate);
       end
       
       function name = get_name()
           name = "dropout_layer";
       end
    end
end
