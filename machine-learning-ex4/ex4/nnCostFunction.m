function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


x = [ones(m,1) X];
h = sigmoid(z3 = ((a2=[ones(m,1) sigmoid(z2=(x*Theta1'))])*Theta2'));

Yy = eye(num_labels);
for i=1:m
	Y = Yy(y(i),:);
	for j=1:num_labels
		J = J+(Y(j)*log(h(i,j)) + (1-Y(j))*log(1-h(i,j)));
	end
end
J = -1*J/m;

t1 =0;t2=0;
for j=1:hidden_layer_size
	for k = 2:input_layer_size+1
		t1 = t1+ Theta1(j,k)^2;
	end
end

for j=1:num_labels
	for k=2:hidden_layer_size+1
		t2= t2+Theta2(j,k)^2;
	end
end

J = J + (t1+t2)*(lambda/(2*m));

d3 = zeros(m,num_labels);
for t=1:m
	d3(t,:) = h(t,:) - Yy(y(t),:);%m X num_labels(k)
end

theta2 = Theta2(:,2:end);
d2 = (d3*theta2).*sigmoidGradient(z2);%z2 -- 5000 x 25

%d2 = d2(:,2:end);

Theta1_grad = (Theta1_grad+ d2'*x)./m;
Theta2_grad = (Theta2_grad + d3'*a2)./m;

theta1 =[zeros(size(Theta1,1),1) Theta1(:,2:end)];
theta2 = [zeros(size(Theta2,1),1) theta2];

Theta1_grad = Theta1_grad.+((lambda/m).*theta1);
Theta2_grad = Theta2_grad.+((lambda/m).*theta2);








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
