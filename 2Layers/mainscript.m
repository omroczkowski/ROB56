% implement actf & check it on a graph%x = -5:0.1:5;%figure(1)%plot(x, actf(x))% implement actdf % note that input value for actdf is not x itself but actf(x)%figure(2)%plot(x, actdf(actf(x)))% implement backprop % it makes sense to start with a really small datasetload tiny.txttlab = tiny(:,1);tvec = tiny(:,2:end);[hlnn1 hlnn2 olnn] = crann(columns(tvec), 8, 4, 2);[size(hlnn1) size(hlnn2) size(olnn)];clsRes = anncls(tvec, hlnn1, hlnn2, olnn);cfmx = confMx(tlab, clsRes)errcf = compErrors(cfmx);for i = 1:10[hlnn1 hlnn2 olnn terr] = backprop(tvec, tlab, hlnn1, hlnn2, olnn, 0.5);endclsRes = anncls(tvec, hlnn1, hlnn2, olnn);cfmx = confMx(tlab, clsRes)errcf = compErrors(cfmx);% now you can (probably) play with ann_training