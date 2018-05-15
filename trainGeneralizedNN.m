function [theta] = trainGeneralizedNN(X, Y, model, theta_init, lambda, iter, batchSize)

    %reshape X according to model(0, :)
    m = size(X, 2);
    X = reshape(X, model(1,4), model(1,4), model(1,3), m);
    batchNo = floor(m/batchSize);
    theta = theta_init; 

    for j = 1:iter
	[X, Y] = orderShuffling(X, Y);
        for i = 1:batchNo
            Xbatch = X(:,:,:, batchSize*(i-1)+1:batchSize*i);
            Ybatch = Y(:, batchSize*(i-1)+1:batchSize*i);
            costFunction = @(t) generalizedCNN(Xbatch, Ybatch, model, t);
            options = optimset('MaxIter', 1, 'GradObj', 'on');
            theta = fmincg(costFunction, theta, options);
        end
        fprintf('Epoch %f finished.\n', j);
    end
end
