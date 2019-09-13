function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
x = X(:, 2);
su1 = 0;

  a = X * theta;
  h = sigmoid(a);
  
  %hx = [log(h), log((1 .- h))];
  %yx = [y, (1 .- y)];
  %yx2 = -1 .* yx;
  %J = sum(yx2 * hx')/m;
  
  for i = 1 : m
    
    su1 = su1 + (- (y(i) * log(h(i))) - ((1 - y(i)) * log( 1 - h(i))));
    
  endfor
  
  J = su1/m;
  
  
  
  
  %grad = sum((h - y) .* x)/m;
  
  for i = 1 : size(theta)
    
    grad(i) = sum((h - y) .* X(:, i))/m;
    
  endfor
  
  

% You need to return the following variables correctly 



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
