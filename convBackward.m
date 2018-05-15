function [boutput, dtheta] = convBackward(binput, model, input, theta)
%   output = CONVFORWARD(input, model, theta) works as backward 
%   convolution layer. 
%   the model is structure of this conv layer. 

    iw = (size(input, 1));
    ic = size(input, 3);
    m = size(input, 4);
    fw = model(2);
    fn = model(3);
    ow = iw;

    %%% 1. calc dtheta %%%

    %reshape input 
    p = floor(fw/2); 
    inputPad = zeros(iw + 2*p, iw + 2*p, ic, m);
    inputPad( p+1 : end-p, p+1 : end-p, :,:) = input(:,:,:,:);
    inputCol = im2col(inputPad, [fw fw ic]);
    inputCol = [ones(1, size(inputCol, 2)); inputCol]; %add bias

    %reshape binput
    a = permute(binput, [3, 1, 2, 4]);
    binputCol = reshape(a, fn, iw^2 * m);

    %back propagation to calc dtheta
    dtheta = (inputCol * binputCol')/m;
    dtheta = dtheta(:);

    %%% 2. calc doutput %%%

    %im2col of binput
    binputPad = zeros(iw + 2*p, iw + 2*p, fn, m);
    binputPad( p+1 : end-p, p+1 : end-p ,:,:) = binput(:,:,:,:);
    binputCol2 = im2col(binputPad, [fw fw fn]);

    %create backward filter by transposing forward filter
    filterCol = reshape(theta, fw^2*ic+1, fn);
    a = filterCol(2:end, :);
    filter = reshape(a, fw, fw, ic, fn);
    bfilter = permute(filter, [1 2 4 3]);
    bfilterCol = reshape(bfilter, fw^2*fn, ic)';

    %back propagation to calc boutput
    boutputCol2 = bfilterCol * binputCol2;
    boutputCol2 = reshape(boutputCol2, ic, iw, iw, m);
    boutput = permute(boutputCol2, [2 3 1 4]);
    
end
