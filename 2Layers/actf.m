function res = actf(tact)
% sigmoid activation function
% tact - total activation
	%columns(tact);
	res = ones(rows(tact),columns(tact));
	for i=1:rows(tact)
		for j=1:columns(tact)
			res(i,j) = 2 / (1 + exp( -tact(i,j) ) ) - 1;
		end
	end

