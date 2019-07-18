clear all;
% clc;
% format compact;
% 
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
x = [train_class_2(:,2:31); train_class_4(:,2:31)]';
x = normalize(x, 'range');

% Matrix/Vector of Target Outputs.
t = [train_class_2(:,1); train_class_4(:,1)]';
t = normalize(t, 'range');

% Testing Data
test_class_2 = preclass{1}(149:212,:);
test_class_4 = preclass{2}(257:357,:);
test_x = [test_class_2(:,2:31); test_class_4(:,2:31)]';
test_x = normalize(test_x, 'range');
test_t = [test_class_2(:,1); test_class_4(:,1)]';
test_t = normalize(test_t, 'range');

%% Preparing and Traing NN.
n = 2;  % the Number of Neurons in the Hidden Layer or [3 2] for 2 hidden layers with 3 and 2 neurons in each layer.
% The neural network is generated as:
net = newff(x,t,2,{'tansig','tansig'}); % Activation functions: tansig (Sigmoid fun) or purelin.
%we need to compare the results if we use purelin instead of tansig because
%we are using tansig function 

% % The NN is initialized with:
net = init(net);
net = configure(net,x,t);
view(net); %it shows the net

% NN training:
net = train(net,x,t);

getwb(net)




%% Calculate the MSE of model with both training and test data

% The output of neural network with the testing data.
est_t = net(x);
plot(t,est_t)
% The MSE criterion can be calculated as:
mse_calc = sum((est_t - t).^2) / length(est_t);
figure()
plotconfusion(t,est_t)
title("Training Confusion Matrix")

%%Calculate the MSE of model with testing data

test_error = test_t - net(test_x);
test_mse_calc = sum((test_error).^2) / length(test_t);
figure()
plotconfusion(test_t,net(test_x));
title("Test Confusion Matrix")