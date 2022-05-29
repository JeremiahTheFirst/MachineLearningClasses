function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X * theta);
left = -y' * log(h);
right = (1-y)' * log(1-h);
sum = left - right;
scaled = sum * (1/m);
theta(1) = 0;
theta_sq = theta' * theta;
regular = (lambda/(2*m)) * theta_sq;
%More mathematic implementation below for fun - hate it though
%J = ((1/m)*((-y' * log(h))-((1-y)' * log(1-h)))) + ((lambda/(2*m)) * theta_sq);
J = scaled + regular;
grad_reg = (lambda/m) * theta
grad = (1/m) * (X' * (h-y)) + grad_reg;

% =============================================================

end
