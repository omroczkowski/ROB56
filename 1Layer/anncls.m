function lab = anncls(tset, hidlw, outlw)
% simple ANN classifier
% tset - data to be classified (every row represents a sample) 
% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix

% lab - classification result (index of output layer neuron with highest value)
% ATTENTION: we assume that constant value IS NOT INCLUDED in tset rows
	tset(1,:);
	hlact = [tset ones(rows(tset), 1)] * hidlw;
	hlact(1,:);
	hlout = actf(hlact);
	hlout(1,:);

	olact = [hlout ones(rows(hlout), 1)] * outlw;
	olact(1,:);
	olout = actf(olact);
	olout(1,:);
	[~, lab] = max(olout, [], 2);
