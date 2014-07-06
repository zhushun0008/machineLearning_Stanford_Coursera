function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vector = [0.01;0.03;0.1;0.3;1;3;10;30];
sigma_vector = [0.01;0.03;0.1;0.3;1;3;10;30];
C_length = length(C_vector);
sigma_length = length(sigma_vector);
crossVal_erro = zeros(C_length,sigma_length);
for i = 1:C_length
	C = C_vector(i);
	for j = 1:sigma_length	
		sigma = sigma_vector(j);
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		pred = svmPredict(model, Xval);
		crossVal_erro(i,j) = mean(double(pred~=yval));
	end
end	
disp(crossVal_erro);
[min_C row] = min(crossVal_erro);
[min_sigma col] = min(min_C);
C = C_vector(row(col))
sigma = sigma_vector(col);


% =========================================================================

end
