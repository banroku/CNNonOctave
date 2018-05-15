function output = convForward(input, model, theta)
%   output = CONVFORWARD(input, model, theta) works as convolution layer. 
%   the model is structure of this conv layer. 

    iw = (size(input, 1));
    ic = size(input, 3);
    m = size(input, 4);
    fw = model(2);
    fn = model(3);
    ow = iw;
    output = zeros(ow, ow, fn, m);

    %create filter
    filterCol = (reshape(theta, fw^2 * ic + 1, fn))';

    %padding
    p = floor(fw/2);
    inputPad = zeros(iw + 2*p, iw + 2*p, ic, m);
    inputPad( p+1 : end-p, p+1 : end-p ,:,:) = input(:,:,:,:);

       
    %im2col
    inputCol = im2col(inputPad, [fw fw ic]);
    inputCol = [ones(1, size(inputCol, 2)); inputCol]; %add bias
    outputCol = filterCol * inputCol;
    output = reshape(outputCol, [fn, ow, ow, m]);
    output = permute(output, [2, 3, 1, 4]);
    
end
