function y = sigmoidGradient(x)
%% calculate sigmoid gradient
    y = sigmoid(x) .* (ones(size(x)) - sigmoid(x));
