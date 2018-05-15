function [Theta, model] = createTheta(model)
%   theta{} = CREATETHETA(model) generates the theta for each layer
%   of the CNN structure designated by model

    k = size(model, 1);     %number of layers   

    theta = cell(k, 1); %array of theta
    theta{1} = [];
    Theta = [];

    for i = 2:k
        if model(i,1) == 1  %conv layer
            a = (model(i-1, 3) * (model(i, 2)^2) + 1) * model(i,3);
            theta{i} = zeros(a, 1);
            theta{i} = randomInitHe(theta{i}, model(i-1, 4), model(i-1, 3));
            model(i,4) = model(i-1, 4);

        elseif model(i,1) == 2  %pooling layer
            theta{i} = [];
            model(i,4) = floor(model(i-1, 4) / model(i, 2));

        elseif model(i,1) == 3  %affine layer
            a = (model(i-1, 4) ^2 * model(i-1, 3) + 1) * model(i, 3);
            theta{i} = zeros(a, 1);
            theta{i} = randomInitHe(theta{i}, model(i-1, 4), model(i-1, 3));
            model(i,2) = 0;
            model(i,4) = 1;

        elseif model(i,1) == 4  %ReLU layer
            theta{i} = [];
            model(i,2) = 0;
            model(i,3) = model(i-1, 3);
            model(i,4) = model(i-1, 4);
        end
    end

    for i = 1:k
        Theta = [Theta; theta{i}(:)];
    end
end
