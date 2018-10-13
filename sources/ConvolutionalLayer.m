classdef ConvolutionalLayer < Layer
    properties(Access = private)
        input_size
        output_size
        kernel_size
        kernel_count
        kernels
    end
    
    methods(Access = public)
        function self = ConvolutionalLayer(input_size, output_size, kernel_size, kernel_count)
            self.input_size = input_size;
            self.output_size = output_size;
            self.kernel_size = kernel_size;
            self.kernel_count = kernel_count;
            self.kernel = rand(kernel_size, kernel_size, kernel_count);
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            self.output_ = convn(input, kernel, 'same');
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, learning_rate)
            % to be completed...
        end
        
        function is = get_input_size(self)
            is = self.input_size;
        end
        
        function os = get_output_size(self)
            os = self.output_size;
        end
    end
end