function output = affineForward(input, model, theta)
%   output = AFFINEFORWARD(input, model, theta) works as affine layer. 
%   The outputs from each node are filled in channel. 

    iw = size(input, 1);
    ic = size(input, 3);
    m = size(input, 4);
    oc = model(3);
    output = zeros(1, 1, oc, m);
    %reshape theta
    input = reshape(input, iw^2 * ic, m);
    input = [ones(1, size(input, 2)); input];
    theta = reshape(theta, oc, iw^2 * ic + 1);
    %reshape input and add bias
    output = theta * input;
    output = reshape(output, 1, 1, oc, m);
end
