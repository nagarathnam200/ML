function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
tx = theta' * X';
sigX = sigmoid(tx);
tmp1 = log(sigX);
tmp1 = -y' .* tmp1;
tmp2 = log(1-sigX);
ym = (1 - y);
tmp2 = ym' .* tmp2;
inner = tmp1 - tmp2;
J = sum(inner(:));
J = J/m;
tmp1 = sigX - y'; 

for i = 1:rows(theta)
  tmp = X(:,i);
  tmp = tmp';
  tmp = tmp .* tmp1;
  grad(i, 1) = (sum(tmp(:)))/m;
endfor

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
