# CNN implementation from scratch on octave
This project is Convolutional Neural Network(CNN) implementation  master and dev are entirely different commit histories. git -ffrom scratch. 
Test data is MNIST.  

This porject is written by octave as programing language, 
since this is inspired by coursera machine learning course 
by Prof Andrew Ng. 

Main purpose of this full scratch CNN implementation is 
to learning basics of neural network and programing by myself. 

This project would not be update any longer 
since the aim was already acheived. 

started from early March, 2018

ended around middle April, 2018


# Brief introduction of usage
There are two scirpts to conduct CNN training & validation. 
- setEnvironment.m
- learnByGeneralizedCNN.m

Firstly, run setEnvironment on your octave, to load mnist dataset and set CNN model. 

Then, run learnByGeneralizedCNN.m to train CNN. 
After the training, it returns cost, accuracy for train/cv sets and also training time. 


1. First run setEnvironment on your octave. 
It loads mnist dataset and set several variables. 
And it imports image package (to use im2col function in it). 
The image package could be downloaded from octave-forge. 

    https://octave.sourceforge.io/packages.php

CNN model could be customized by change model in this setEnvironment.m. 

> %model of CNN
> %col1: type of layer. 0=input, 1=conv, 2=pool, 3=affine, 4=ReLU. 
> %col2: filter size for conv/pooling layer (any number for other layers)
> %col3: channel(filter) number for input/conv layer
> %      output number for affine layer
> %col4: image width of input (any number for other layers)
> 
> k = 3;
> model = ...
> [0  0  2  4
>  1  3  2  4
>  4  0  2  4
>  2  2  2  2
>  3  0 10  1 ];

Initial model shown below is very simple. 

> Convolutional -> ReLU -> Pooling -> Affine

Output layer is softmax.

2. Then run learnByGeneralizedCNN.m
Here use stochastic gradient desent and parameters are following: 
- batchSize: batchSize. 
- iter: Epoch. 
- lambda: regularization parameter (!not implemented)
- initialzeTheta: randomly initialze weight if Ture. 
                Else use weight trained by previous training. 


# Something not implemented. 
1. Regulaziation
It could be implemented by adding regularization term in 
cost and gradient calculation with lambda...

# Caution
The caluculation use much of your machine power. 
PC could freeze if you will try large scale test, especially: 
- large model(deep layer, much channel, large conv. filter etc.)
- large batch scale

