function boutput = poolBackward(binput, model, input, index)
%   boutput = poolBackward(binput, model, index) works as backward 
%   pooling layer. 
%   This needs poolingindex generated on forward pooling, 
%   other than backward input and model. 

    iw = (size(input, 1));
    ic = size(input, 3);
    m = size(input, 4);
    fw = model(2);
    %reshape binput
    binputCol = binput(:)';
    boutputCol = binputCol .* index;
    boutput = col2im(boutputCol, [fw fw], [iw iw ic m], "distinct");

end
