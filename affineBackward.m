function [boutput, dtheta] = affineBackward(binput, model, input, theta)
%   [boutput, dtheta] = AFFINEBACKWARD(binput, output, theta) works as 
%   backward affine layer with backward signal BINPUT and output/theta 
%   of the layer. 
%   This returns backward output BOUTPUT and gradient of theta DTHETA. 
    iw = size(input, 1);
    ic = size(input, 3);
    m = size(input, 4);
    oc = model(3);
    
    %reshape 
    input = reshape(input, iw^2 * ic, m);
    input = [ones(1, size(input, 2)); input];   %add bias
    theta = reshape(theta, oc, iw^2 * ic + 1);
    binputCol = reshape(binput, 1^2 * model(3), m);

    dtheta = (binputCol * input')/m;
    boutputCol = theta' * binputCol;
    boutputCol = boutputCol(2:end, :);

    %reshape boutput
    boutput = reshape(boutputCol, iw, iw, ic, m);
    dtheta = dtheta(:);
end
