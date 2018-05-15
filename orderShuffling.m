function [X_sh Y_sh] = orderShuffling(X,Y)
    index = randperm(size(X,4));
    X_sh = X(:,:,:, index);
    Y_sh = Y(:, index);
end    
