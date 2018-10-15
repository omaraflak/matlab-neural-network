classdef ConvolutionalLayer < Layer
    properties(Access = private)
        kernel_size
        layer_depth
        input_depth
        weights
        bias
    end
    
    methods(Access = public)
        function self = ConvolutionalLayer(input_size, kernel_size, layer_depth)            
            if length(input_size)==2
                self.input_depth = 1;
                self.output_size = input_size - kernel_size + 1;
            else
                self.input_depth = input_size(3);
                self.output_size = [input_size(1)-kernel_size(1)+1 input_size(2)-kernel_size(2)+1 input_size(3)];
            end
            
            self.input_size = input_size;
            self.kernel_size = kernel_size;
            self.layer_depth = layer_depth;
            self.weights = rand([kernel_size layer_depth]);
            self.bias = rand([self.output_size layer_depth]);
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            self.output_ = zeros([self.output_size(1:2) self.input_depth*self.layer_depth]);
            index = 1;            
            for d=1:self.input_depth
                for k=1:self.layer_depth
                    self.output_(:,:,index) = self.cross_corr2(input(:,:,d), self.weights(:,:,k)) + self.bias(:,:,k);
                    index = index + 1;
                end
            end
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, learning_rate)
            in_error = zeros([self.input_size self.layer_depth]);            
            dWeights = zeros([self.kernel_size self.input_depth*self.layer_depth]);
            index = 1;
            for d=1:self.input_depth
                for k=1:self.layer_depth
                    in_error(:,:,index) = conv2(error(:,:,k), self.weights(:,:,k));
                    dWeights(:,:,index) = self.cross_corr2(self.input_(:,:,d), error(:,:,k));
                    index = index + 1;
                end
            end

            self.weights = self.weights - learning_rate*dWeights;
            self.bias = self.bias - learning_rate*error; % dBias = error
        end
        
        function block = save(self)
            block{1} = self.input_size;
            block{2} = self.kernel_size;
            block{3} = self.layer_depth;
            block{4} = self.weights;
            block{5} = self.bias;
        end
    end
    
    methods(Static)
        function layer = load(block)
            input_size = block{1};
            kernel_size = block{2};
            layer_depth = block{3};
            layer = ConvolutionalLayer(input_size, kernel_size, layer_depth);
            layer.weights = block{4};
            layer.bias = block{5};
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