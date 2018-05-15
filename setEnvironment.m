% === Reset and Set-up environment to start mnist project
%
%
%
%

clear all; close all; clc
load "mymnist_trainingData.mat";
load "mymnist_cvAndTestData.mat";
load "mnist_mini.mat";
pkg load image;
initializeTheta = true;
K=10;

%Transpose Xs
Xtrain = Xtrain'; Xcv = Xcv'; Xtest = Xtest';

%convert 0 to 10 in Y
Ytrain = Ytrain + (Ytrain==0)*10;
Ycv = Ycv + (Ycv==0)*10;
Ytest = Ytest + (Ytest==0)*10;

%make 01 matrix of y
Ytrain = matrixizeY(Ytrain, K);
Ycv = matrixizeY(Ycv, K);
Ytest = matrixizeY(Ytest, K);

%check row-column true 
if (checkRC(Xtrain, Xcv) == false)
    fprintf('Row and column of X maybe conversed.');
endif
if (checkRC(Ytrain, Ycv) == false)
    fprintf('Row and column of Y maybe conversed.');
endif

%feature scaling
[Xtrain Xcv Xtest mu range] = featureScaling(Xtrain, Xcv, Xtest);

%model of CNN
%col1: type of layer. 0=input, 1=conv, 2=pool, 3=affine, 4=ReLU. 
%col2: filter size for conv/pooling layer (any number for other layers)
%col3: channel(filter) number for input/conv layer
%      output number for affine layer
%col4: image width of input (any number for other layers)

k = 3;
model = ...
[0  0  2  4
 1  3  2  4
 4  0  2  4
 2  2  2  2
 3  0 10  1 ];

%other useful constants
m = size(Xtrain, 2);
n = size(Xtrain, 1);
%end
