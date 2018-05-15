function [Xtrain_sc Xcv_sc Xtest_sc mu range] = featureScaling(Xtrain, Xcv, Xtest)
    
    range = 255;
    Xtrain_sc = double(Xtrain) / range;
    Xcv_sc = double(Xcv) / range;
    Xtest_sc = double(Xtest) / range;
    
%    % mu = 0.5?
%    mu = mean(mean(Xtrain_sc));
%    Xtrain_sc = Xtrain_sc - mu;
%    Xcv_sc = Xcv_sc - mu;
%    Xtest_sc = Xtest_sc - mu;
    mu = mean(mean(Xtrain_sc));

end    
