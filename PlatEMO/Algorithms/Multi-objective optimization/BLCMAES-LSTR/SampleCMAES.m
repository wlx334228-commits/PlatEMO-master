function [Dec,Step] = SampleCMAES(CMA,N)
% Sample bounded solutions from the current CMA-ES distribution.

    if nargin < 2 || isempty(N)
        N = CMA.lambda;
    end

    cholC = SafeChol(CMA.C);
    Step = randn(N,CMA.D) * cholC;
    Dec = repmat(CMA.xmean,N,1) + Step .* repmat(CMA.sigma,N,1);
    Dec = RepairBounds(Dec,CMA.lower,CMA.upper);
    Step = CMAESStepsFromDec(CMA,Dec);
end
