function lab = anncls(tset, hidlw1, hidlw2, outlw)
% simple ANN classifier
% tset - data to be classified (every row represents a sample) 
% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix

% lab - classification result (index of output layer neuron with highest value)
% ATTENTION: we assume that constant value IS NOT INCLUDED in tset rows

	hlact1 = [tset ones(rows(tset), 1)] * hidlw1;
	hlout1 = actf(hlact1);

	hlact2 = [hlout1 ones(rows(hlout1), 1)] * hidlw2;
	hlout2 = actf(hlact2);

	olact = [hlout2 ones(rows(hlout2), 1)] * outlw;
	olout = actf(olact);
	[~, lab] = max(olout, [], 2);
