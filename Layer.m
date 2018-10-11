classdef (Abstract) Layer < handle
    properties(Access = protected)
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
    end
end