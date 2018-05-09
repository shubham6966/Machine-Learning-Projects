function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

Ctest = 0.003;
sigmaTest = 0.01;
Cget= 0;
sigmaget = 0;
error = 100000;
for i=1:8
	Ctest = Ctest*3; 
	for j=1:8
		sigmaTest = sigmaTest*3;
		model= svmTrain(X, y, Ctest, @(x1, x2) gaussianKernel(x1, x2, sigmaTest));
		predictions = svmPredict(model,Xval);
		predictions = mean(double(predictions~=yval));
		if(error>predictions)
			error = predictions;
			Cget = Ctest;
			sigmaget = sigmaTest;
		end
	end
	sigmaTest = 0.01;
end
C = Cget;
sigma = sigmaget;








% =========================================================================

end
