function [J dTheta] = generalizedCNN(input, label, model, Theta)
%function [J dtheta dtheta_math] = generalizedCNN(input, label, model, theta)
%   [J grad] = GERALIZEDCNN(X, Y, model, theta) create CNN according to 
%   the model and works forward propagation and backward propagation to 
%   return cost J and gradient grad. 
%   input should be in demenstion of [width, width, channel, m].
%   model should be 2D matrix (see setEnvironment.m). 
%   theta should be array of matrix created by createTheta(model). 

    J = 0;

    k = size(model, 1); 
    m = size(input, 4);
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

    binput = reshape((y - label), 1, 1, model(k,3), m);  %backward input for test
    boutput{k+1} = binput;

    dtheta = cell(k,1);
    for i = 1:k
        dtheta{i} = zeros(size(theta{i}));
    end

    % Backward propagation to calculate gradient of theta 
    for i = [k:-1:2]
	    if model(i, 1) == 1 %conv
            [boutput{i}, dtheta{i}] = ...
                convBackward(boutput{i+1}, model(i,:), output{i-1}, theta{i});

        elseif model(i, 1) == 2 %pool
            boutput{i} = ...
                poolBackward(boutput{i+1}, model(i,:), ...
                                output{i-1}, poolindex{i});
            dtheta{i} = [];

        elseif model(i, 1) == 3 %affine
            [boutput{i}, dtheta{i}] = ... 
                affineBackward(boutput{i+1}, model(i,:), ...
                                output{i-1}, theta{i});

        elseif model(i, 1) == 4 %ReLU
            boutput{i} = reluBackward(boutput{i+1}, output{i-1});
            dtheta{i} = [];
	    end
    end 
    
    dTheta = [];
    for i = 1:k
        dTheta = [dTheta; dtheta{i}(:)];
    end

    %% %==gradient checking==
    %% dtheta_math = dtheta;
    %% for i = 1:k
    %%     dtheta_math{i}(:) = 0;
    %% end
    %% outputPlus = cell(k, 1);
    %% outputPlus{1} = input;
    %% delta = 0.0001;

    %% for l = 2:5 
    %%     for j = 1:size(theta{l})
    %%         thetaPlus = theta;
    %%         thetaPlus{l}(j) = thetaPlus{l}(j) + delta;
    %%                
    %%         % Forward propagation to calculate output from input
    %%         for i = 2:k
    %%     	    if model(i, 1) == 1 %conv
    %%                 outputPlus{i} = convForward(outputPlus{i-1}, model(i,:), thetaPlus{i});
    %%                 
    %%             elseif model(i, 1) == 2 %pool
    %%                 [outputPlus{i} poolindex{i}] = poolForward(outputPlus{i-1}, model(i,:));
    %%     
    %%             elseif model(i, 1) == 3 %affine
    %%                 outputPlus{i} = affineForward(outputPlus{i-1}, model(i,:), thetaPlus{i});

    %%             elseif model(i, 1) == 4 %ReLU
    %%                 outputPlus{i} = reluForward(outputPlus{i-1});
    %%     
    %%     	    end
    %%         end 
    %%     
    %%         % Calculate Cost J (cross entropy?)
    %%         yPlus = softmax(outputPlus{k});
    %%         JPlus = - sum(sum(label .* log(yPlus)))/m;
    %%         dtheta_math{l}(j) = (JPlus - J)/delta;
    %%     end
    %% end
end
