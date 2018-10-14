classdef MaxPoolLayer < Layer
    properties(Access = private)
        kernel_size
    end
    
    methods(Access = public)
        function self = MaxPoolLayer(input_size, kernel_size)
            self.input_size = input_size;
            self.output_size = input_size - kernel_size + 1;
            self.kernel_size = kernel_size;
        end
        
        function output = forward_propagation(self, input)
            self.input_ = input;
            self.output_ = zeros(self.output_size);
            for k=1:self.input_size(3)
                self.output_(:,:,k) = self.maxpool(input(:,:,k), self.kernel_size);
            end
            output = self.output_;
        end
        
        function in_error = back_propagation(self, error, ~)
            in_error = zeros(self.input_size);
            for k=1:self.input_size(3)
                in_error(:,:,k) = self.maxpool_back(self.input_(:,:,k), self.kernel_size, error(:,:,k));
            end
        end
        
        function block = save(self)
            block{1} = self.input_size;
            block{2} = self.kernel_size;
        end
    end
    
    methods(Static)
        function layer = load(block)
            input_size = block{1};
            kernel_size = block{2};
            layer = MaxPoolLayer(input_size, kernel_size);
        end
        
        function name = get_name()
            name = "maxpool_layer";
        end
    end
    
    methods(Access = private, Static)
        function c = maxpool(a, b)
            new_height = size(a,1)-b(1)+1;
            new_width = size(a,2)-b(2)+1;
            c = zeros(new_height, new_width);
            for i=1:new_height
                for j=1:new_width
                    m = a(i,j);
                    for u=1:b(1)
                        for v=1:b(2)
                            m = max(m, a(i+u-1, j+v-1));
                        end
                    end
                    c(i,j) = m;
                end
            end
        end
        
        function c = maxpool_back(a, b, error)
            c = zeros(size(a));
            for i=1:size(a,1)-b(1)+1
                for j=1:size(a,2)-b(2)+1
                    m = a(i,j);
                    XY = [i j];
                    for u=1:b(1)
                        for v=1:b(2)
                            if m < a(i+u-1, j+v-1)
                                m = a(i+u-1, j+v-1);
                                XY = [i+u-1 j+v-1];
                            end
                        end
                    end
                    c(XY(1), XY(2)) = error(i,j);
                end
            end
        end
    end
end