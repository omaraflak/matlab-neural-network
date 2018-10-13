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