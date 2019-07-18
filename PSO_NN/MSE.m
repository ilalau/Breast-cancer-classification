function MSE_calc = MSE( wb, net, inputs, targets)

% wb is the weights and biases row vector
% It must be transposed when transferring the weights and biases to the network net.

 net = setwb(net, wb');
 
 yest = net(inputs);
 
 MSE_calc = sum((yest-targets).^2) / length(yest);