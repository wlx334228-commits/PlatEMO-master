function Step = CMAESStepsFromDec(CMA,Dec)
% Convert bounded decisions into normalized CMA-ES steps.

    safeSigma = CMA.sigma;
    safeSigma(safeSigma < 1e-12) = 1e-12;
    Step = (Dec - repmat(CMA.xmean,size(Dec,1),1)) ./ repmat(safeSigma,size(Dec,1),1);
end
