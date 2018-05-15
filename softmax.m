function output = softmax(input)
%   output = SOFTMAX(input)

    iw = size(input, 1) * size(input, 2);
    if iw > 1 
        fprintf('Error: Output width of final layer over 1\n');
    end

    ic = size(input, 3);
    m = size(input, 4);

    input = reshape(input, ic, m);
    inputMax = max(input);
    input = input - inputMax;
    output = exp(input) ./ sum(exp(input));

end
