clear all;
clc;
% format compact;

%% Data preparation: Split into train data and test data.
load('data.mat');
T = data(:,1);

preclass = cell(2,1) ;
class = [2 4] ;
for i = 1:2
    preclass{i} = data(T==class(i),:) ;
end

% Training Data
train_class_2 = preclass{1}(1:148,:);
train_class_4 = preclass{2}(1:256,:);

% Matrix of Inputs
inputs = [train_class_2(:,2:31); train_class_4(:,2:31)]';
inputs = normalize(inputs, 'range');

% Matrix/Vector of Target Outputs.
targets = [train_class_2(:,1); train_class_4(:,1)]';
targets = normalize(targets, 'range');

% Testing Data
test_class_2 = preclass{1}(149:212,:);
test_class_4 = preclass{2}(257:357,:);
test_x = [test_class_2(:,2:31); test_class_4(:,2:31)]';
test_x = normalize(test_x, 'range');
test_t = [test_class_2(:,1); test_class_4(:,1)]';
test_t = normalize(test_t, 'range');

[numVar, numSamp] = size(inputs);
%% Initialize the NN
% % Number of neurons
n = 2;

% The neural network is generated as:
net = newff(inputs,targets,2,{'tansig','tansig'}); % Activation functions: tansig (Sigmoid fun) or purelin.

% The NN is initialized with:
% net = init(net);
net = configure(net,inputs,targets);
view(net);

%% Apply PSO to NN - Training
% calculates MSE
h = @(wb) MSE(wb, net, inputs, targets); % MSE - cost function independent on weight-bias.

% running the particle swarm optimization algorithm with desired options
[sol, err_ga] = PSO(h, (numVar+1)*n +n + 1);
net = setwb(net, sol');
% 
% error MSE PSO optimized NN
error = targets - net(inputs);
MSE_calc = sum((error).^2) / length(targets)
figure()
plotconfusion(targets,net(inputs))
title("Training Confusion Matrix Weighted PSO")

%% Testing
test_error = test_t - net(test_x);
test_MSE = sum((test_error).^2) / length(test_t)

figure()
plotconfusion(test_t,net(test_x));
title("Test Confusion Matrix Weighted PSO")