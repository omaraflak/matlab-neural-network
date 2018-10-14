classdef ConvolutionalLayer < Layer
    properties(Access = private)
        kernel_size
        kernel_count
        weights
    end
    
    methods(Access = public)
        function self = ConvolutionalLayer(input_size, kernel_size, kernel_count)
            self.input_size = input_size;
            self.output_size = input_size - kernel_size + 1;
            self.kernel_size = kernel_size;
            self.kernel_count = kernel_count;
            self.weights = rand(kernel_size);
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            self.output_ = self.cross_corr2(input, self.weights);
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, learning_rate)
            dWeights = self.cross_corr2(self.input_, error);
            self.weights = self.weights - learning_rate*dWeights;
            in_error = xconv2(error, self.input_);
        end
        
        function block = save(self)
        end
    end
    
    methods(Static)
        function layer = load(block)
        end
        
        function name = get_name()
            name = "convolutional_layer";
        end
    end
    
    methods(Access = private, Static)
        function c = cross_corr2(a, b)
            c = xcorr2(a,b);
            c = c(size(b,2):size(c,2)-size(b,2)+1, size(b,1):size(c,1)-size(b,1)+1);
        end
    end
end