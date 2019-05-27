function [hl1 hl2 ol] = crann(cfeat, chn1, chn2, cclass)
% generates hidden and output ANN weight matrices
% cfeat - number of features 
% chn1 - number of neurons in the 1 hidden layer
% chn2 - number of neurons in the 2 hidden layer
% cclass - number of neurons in the outpur layer (= number of classes)

% hl1 - hidden layer weight matrix
% hl2 - hidden layer weight matrix
% ol - output layer weight matrix

% ATTENTION: we assume that constant value (bias) IS NOT INCLUDED

	hl1 = (rand(cfeat + 1, chn1) - 0.5) / sqrt(cfeat + 1);
	hl2 = (rand(chn1 + 1, chn2) - 0.5) / sqrt(chn1 + 1);
	ol = (rand(chn2 + 1, cclass) - 0.5) / sqrt(chn2 + 1);
