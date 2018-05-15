function model = modelCheck(protoModel)
%   model = modelCheck(X, protoModel) calculate output size of each layer. 

    model = [model zeros(k, 1)]; %add col to fill output size
    model(1, 4) = model(1, 2)^2;
 
    %input 
    for i = 2:k
        if model(i,1) == 1  %conv layer
            model(i,4) = model(i-1, 4);

        elseif model(i,1) == 2  %pooling layer
            model(i,4) = floor(model(i-1, 4)/model(i, 2));
    
        elseif model(i,1) == 3
            model(i,4) = model(i-1, 2);
            model(i,5) = 1;

end
