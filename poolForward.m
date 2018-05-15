function [output index] = poolForward(input, model)
%   [output index] = poolForward(input, model) works as pooling layer. 
%   The model is structure of this pool layer.
%   The index notes position of pooled index. 
    iw = (size(input, 1));
    ic = size(input, 3);
    m = size(input, 4);
    fw = model(2);
    ow = floor(iw/fw);
    output = zeros(ow, ow, ic, m);

    %im2col
    inputCol = im2col(input, [fw fw], "distinct");
    outputCol = max(inputCol);
    output = reshape(outputCol, [ow, ow, ic, m]);
    
    %create pooling index
    index = not(inputCol - outputCol);
    %index= col2im(indexCol, [ow, ow], [iw iw ic m], "distinct");

end
