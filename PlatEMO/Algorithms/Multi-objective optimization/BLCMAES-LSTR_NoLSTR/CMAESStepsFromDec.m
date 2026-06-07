function Step = CMAESStepsFromDec(CMA,Dec)
% Convert bounded decisions into normalized CMA-ES steps.

    Step = (Dec - repmat(CMA.xmean,size(Dec,1),1)) ./ max(CMA.sigma,1e-12);
end
