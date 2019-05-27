function res = actdf(sfvalue)
% derivative of sigmoid activation function
% sfvalue - value of sigmoid activation function (!)

	res = zeros(rows(sfvalue),columns(sfvalue));
	for i=1:rows(sfvalue)
		for j=1:columns(sfvalue)
			res(i,j) = (1-sfvalue(i,j)*sfvalue(i))/2;
		end
	end

