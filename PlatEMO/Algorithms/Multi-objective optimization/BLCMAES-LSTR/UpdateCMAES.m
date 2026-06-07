function CMA = UpdateCMAES(CMA,Step,Dec,Fitness)
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

    selectedDec = Dec(rank(1:useMu),:);
    selectedStep = Step(rank(1:useMu),:);

    xold = CMA.xmean;
    xnew = weights * selectedDec;
    xnew = RepairBounds(xnew,CMA.lower,CMA.upper);

    safeSigma = CMA.sigma;
    safeSigma(safeSigma < 1e-12) = 1e-12;
    meanStep = (xnew - xold) ./ safeSigma;

    cholC = SafeChol(CMA.C);
    CMA.ps = (1-CMA.cs)*CMA.ps + ...
        sqrt(CMA.cs*(2-CMA.cs)*mueff) * (meanStep / cholC');

    sigmaScale = exp(CMA.cs/CMA.ds*(norm(CMA.ps)/CMA.ENN-1))^0.3;
    CMA.sigma = min(max(CMA.sigma .* sigmaScale,CMA.minSigma),CMA.maxSigma);

    CMA.iter = CMA.iter + 1;
    hsig = norm(CMA.ps)/sqrt(1-(1-CMA.cs)^(2*CMA.iter)) < CMA.hth;
    delta = (1-hsig)*CMA.cc*(2-CMA.cc);
    CMA.pc = (1-CMA.cc)*CMA.pc + ...
        hsig*sqrt(CMA.cc*(2-CMA.cc)*mueff)*meanStep;

    CMA.C = (1-CMA.c1-CMA.cmu)*CMA.C + ...
        CMA.c1*(CMA.pc'*CMA.pc + delta*CMA.C);
    for i = 1 : useMu
        CMA.C = CMA.C + CMA.cmu*weights(i)*(selectedStep(i,:)'*selectedStep(i,:));
    end

    CMA.C = RepairCovariance(CMA.C);
    CMA.xmean = xnew;
end
