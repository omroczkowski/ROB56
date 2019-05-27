function [hidlw1 hidlw2 outlw terr] = backprop(tset, tslb, inihidlw1, inihidlw2, inioutlw, lr)
% derivative of sigmoid activation function
% tset - training set (every row represents a sample)
% tslb - column vector of labels 
% inihidlw - initial hidden layer weight matrix
% inioutlw - initial output layer weight matrix
% lr - learning rate

% hidlw - hidden layer weight matrix
% outlw - output layer weight matrix
% terr - total squared error of the ANN

% 1. Set output matrices to initial values
	hidlw1 = inihidlw1;
	hidlw2 = inihidlw2;
	outlw = inioutlw;
	
% 2. Set total error to 0
	terr = 0;
	
% foreach sample in the training set
	for i=1:rows(tset)
		i;
		%outvec = repmat(-1, 1, 2);
		%outvec(tslb(i)) = 1;

		%hlact = [tset(i, :) 1] * hidlw;
		%hlout = actf(hlact);

		%olact = [hlout 1] * outlw;
		%olout = actf(olact);
		
		%outdel = (outvec-olout) .* actdf(olact);
		%hiddel = outdel * outlw(1:(end-1),:)'.*actdf(hlact);
		%outlw = outlw + lr * ([hlout, 1]' * outdel);
		%hidlw = hidlw + lr * ([tset(i,:), 1]' * hiddel);


		% 3. Set desired output of the ANN
		desired_output = zeros(1,columns(outlw))-1;
		desired_output(1,tslb(i))=1;
		
		% 4. Propagate input forward through the ANN
		% remember to extend input [tset(i, :) 1]
		hlact1 = [tset(i, :) 1] * hidlw1;
		hlout1 = actf(hlact1);
	
		hlact2 = [hlout1 1] * hidlw2;
		hlout2 = actf(hlact2);

		olact = [hlout2 1] * outlw;
		olout = actf(olact);

		% 5. Adjust total error (just to know this value)
		err = (desired_output - olout).^2;
		err = sum(err,2);
		terr = terr + err;
		% 6. Compute delta error of the output layer
		% how many delta errors should be computed here?
		#out_delta = (desired_output - olout)/2.*(ones(rows(olout),columns(olout))-olout.^2);
		out_delta = (desired_output - olout).*actdf(olout);

		% 7. Compute delta error of the hidden layer
		% how many delta errors should be computed here?
		%(outlw(1:end-1, :) * out_delta')
		%out_delta * outlw(1:end-1, :)' 
		delta2 = (outlw(1:end-1, :) * out_delta').*actdf(hlout2)';
		delta1 = (hidlw2(1:end-1, :) * delta2).*actdf(hlout1)';

		% 8. Update output layer weights
		outlw = outlw + (lr * (out_delta' * [hlout2 1])');

		
		% 9. Update hidden layer weights
		hidlw2 = hidlw2 + (lr * (delta2 * [hlout1 1])');
		hidlw1 = hidlw1 + (lr * (delta1 * [tset(i, :) 1])');

	end

