function CMA = UpdateCMAES(CMA,Dec,Fitness)
% Update CMA-ES parameters from evaluated candidate solutions.

    if isempty(Dec)
        return;
    end

    Fitness = Fitness(:);
    [~,rank] = sort(Fitness,'ascend');
    useMu = min(CMA.mu,numel(rank));
    weights = CMA.weights(1:useMu);
    weights = weights ./ sum(weights);
    mueff = 1 / sum(weights.^2);

    xold = CMA.xmean;
    X = Dec(rank(1:useMu),:);
    Y = (X - repmat(xold,useMu,1)) ./ max(CMA.sigma,1e-12);
    mahalanobis = sqrt(sum((Y*CMA.invsqrtC').^2,2));
    scale = min(1,CMA.cy./max(mahalanobis,1e-12));
    Y = Y .* repmat(scale,1,CMA.dim);
    meanStep = weights * Y;

    xnew = xold + meanStep * CMA.sigma;
    xnew = RepairBounds(xnew,CMA.lower,CMA.upper);

    CMA.ps = (1-CMA.cs)*CMA.ps + ...
        sqrt(CMA.cs*(2-CMA.cs)*mueff) * meanStep * CMA.invsqrtC;

    deltaSigma = (CMA.cs/CMA.ds)*(norm(CMA.ps)/CMA.ENN-1);
    CMA.sigma = CMA.sigma * exp(min(deltaSigma,CMA.deltaSigmaMax));
    CMA.sigma = min(max(CMA.sigma,CMA.minSigma),CMA.maxSigma);

    CMA.iter = CMA.iter + 1;
    hsig = norm(CMA.ps)/sqrt(1-(1-CMA.cs)^(2*CMA.iter)) < CMA.hth;
    CMA.pc = (1-CMA.cc)*CMA.pc + ...
        hsig*sqrt(CMA.cc*(2-CMA.cc)*mueff)*meanStep;

    Cmu = Y' * diag(weights) * Y;
    CMA.C = (1-CMA.c1-CMA.cmu)*CMA.C + ...
        CMA.c1*(CMA.pc'*CMA.pc) + CMA.cmu*Cmu;
    CMA.C = triu(CMA.C) + triu(CMA.C,1)';
    CMA.xmean = xnew;
    CMA = RepairCMAES(CMA);
end
