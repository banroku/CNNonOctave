function [Acc, J, output] = calculateAccuracy(input, label, model, Theta)
%   Ypredict = predict(X, model, theta) predicts Y from the input
    m = size(input, 2);
    input = reshape(input, model(1,4), model(1,4), model(1,3), m);

    J = 0;
    k = size(model, 1); 
    output = cell(k, 1);
    output{1} = input;
    poolindex = cell(k, 1);
    boutput = cell(k+1,1);

    % Separate theta
    theta = cell(k, 1);
    theta{1} = [];
    sepFrom = 0;

    for i = 2:k
	    if model(i, 1) == 1 %conv
            a = (model(i-1, 3) * (model(i, 2)^2) + 1) * model(i,3);
            sepTo = sepFrom + a;
            theta{i} = Theta(sepFrom+1 : sepTo);
            sepFrom = sepTo;
            
        elseif model(i, 1) == 2 %pool

        elseif model(i, 1) == 3 %affine
            a = (model(i-1, 4) ^2 * model(i-1, 3) + 1) * model(i, 3);
            sepTo = sepFrom + a;
            theta{i} = Theta(sepFrom+1 : sepTo);
            sepFrom = sepTo;

        elseif model(i, 1) == 4 %ReLU

	    end
    end 

    % Forward propagation to calculate output from input
    for i = 2:k
	    if model(i, 1) == 1 %conv
            output{i} = convForward(output{i-1}, model(i,:), theta{i});

        elseif model(i, 1) == 2 %pool
            [output{i} poolindex{i}] = poolForward(output{i-1}, model(i,:));

        elseif model(i, 1) == 3 %affine
            output{i} = affineForward(output{i-1}, model(i,:), theta{i});

        elseif model(i, 1) == 4 %ReLU
            output{i} = reluForward(output{i-1});

	    end
    end 
    % Calculate Cost J (cross entropy?)

    y = softmax(output{k});
    J = - sum(sum(label .* log(y)))/m;

    [a, y] = max(y);
    [a, label] = max(label);
    correctList = (y == label);
    Acc = sum(correctList)/size(correctList, 2);
