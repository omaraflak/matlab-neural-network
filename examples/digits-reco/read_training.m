% The training file consists of 946 digits (0-9) drawn on a 32x32 array
% After each digit, the number represented is written on a separate line

function [x_train, y_train, x_test, y_test] = read_training(filename)
    input = zeros(1, 32*32, 946);
    output = zeros(1, 10, 946);
    
    file = fopen(filename);
    for i = 1:946
        for j = 1:32
            line = fgetl(file) - '0';
            input(1, 32*(j-1)+1 : 32*(j-1)+32, i) = line;
        end
        out = fgetl(file);
        vect = zeros(1,10);
        vect(str2num(out) + 1) = 1;
        output(1,:,i) = vect;
    end
    fclose(file);

    x_train = input(1,:,1:900);
    y_train = output(1,:,1:900);
    x_test = input(1,:,901:946);
    y_test = output(1,:,901:946);
end