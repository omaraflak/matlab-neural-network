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
        
        function block = save(self)
            block{1} = self.input_size;
            block{2} = self.output_size;
            block{3} = self.kernel_size;
            block{4} = self.kernel_count;
            block{5} = self.kernels;
        end
    end
    
    methods(Static)
        function layer = load(block)
            input_size = block{1};
            output_size = block{2};
            kernel_size = block{3};
            kernel_count = block{4};
            layer = ConvolutionalLayer(input_size, output_size, kernel_size, kernel_count);
            layer.kernels = block{5};
        end
        
        function name = get_name()
            name = "convolutional_layer";
        end
    end
end