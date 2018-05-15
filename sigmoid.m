function y = sigmoid(x)
    y = ones(size(x)) ./ (ones(size(x)) + exp(-x));
end
