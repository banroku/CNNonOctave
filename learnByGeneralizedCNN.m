% === Learning By Logistic regression ===
%
%
%
%

J = 0;
lambda = 0.000;
iter = 001;
batchSize = 00005;
initializeTheta = true;

% === architecture of NN ==
%model of CNN
%col1: type of layer. 0=input, 1=conv, 2=pool, 3=affine, 4=ReLU. 
%col2: filter size for conv/pooling layer (any number for other layers)
%col3: channel(filter) number for input/conv layer
%      output number for affine layer
%col4: image width of input (any number for other layers)

bs = batchSize;
model = ...
[ 0  0  1 28
  1  5 10 28
  4  0 10 28
  2  2 10 14
  3  0 10  1 ];

result = [];

for i = 1:1
    
    % make new theta
    if initializeTheta
        [Theta, model] = createTheta(model);
        Theta_init = Theta;
    end
    
    tic();
    % % train by NN
    % theta = trainNN(Xtrain, Ytrain, K1, K2, theta, lambda, iter);
    % train by generalized NN
    Theta = trainGeneralizedNN(Xmini, Ymini, model, Theta, lambda, iter, batchSize);
    
    trainingTime = toc();
    
    % continue using trained theta
    initializeTheta = false;
    
    %hear need to implement prediction and calculateAccuracy
    [Acc_train, J_train, output] = calculateAccuracy(Xmini, Ymini, model, Theta);
    fprintf('J_train, Acc_train, trainingTime = %d, %d, %d\n', J_train, Acc_train, trainingTime);
    result = [result; i Acc_train output{4}(:)'];

end

[Acc_cv, J_cv, output] = calculateAccuracy(Xcv(:,1:1000), Ycv(:,1:1000), model, Theta);
    fprintf('J_cv, Acc_cv = %d, %d\n', J_cv, Acc_cv);

% % calculate parameters of cross-valication set
% J_train = costGeneralizedNN(Xtrain, Ytrain, K, theta, 0);
% J_cv = costGeneralizedNN(Xcv, Ycv, K, theta, 0);
% Pre_cv= predictGeneralizedNN(Xcv, K, theta);
% Acc_cv = calculateAccuracy(Ycv, Pre_cv);
% 
% fprintf('Train time (per iter): %f (%f) \n', trainingTime, trainingTime/iter);
% fprintf('Cost (train, cv): %f, %f \n', J_train, J_cv);
% fprintf('Accuracy (train, cv): %f, %f \n', Acc_train, Acc_cv);
% 
% % calculate costs
