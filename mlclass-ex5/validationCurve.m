function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
lambdaNum = size(lambda_vec,1);
theta_vec = zeros(size(X,2)+1,lambdaNum);
% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
for i = 1:lambdaNum
	lambda = lambda_vec(i);
	theta = trainLinearReg([ones(size(X,1),1) X], y, lambda);
	theta_vec(:,i) = theta;
	error_train(i) = linearRegCostFunction([ones(size(X,1), 1) X], y, theta, 0);
	error_val(i) = linearRegCostFunction([ones(size(Xval,1), 1) Xval], yval, theta, 0);
end
[lambdaMin, index] = min(error_val);









% =========================================================================

end
