function [Dec,Step] = SampleCMAES(CMA,N)
% Sample bounded solutions from the current CMA-ES distribution.

    if nargin < 2 || isempty(N)
        N = CMA.lambda;
    end

    Step = (randn(N,CMA.dim) .* repmat(CMA.eigD,N,1)) * CMA.B';
    Dec = repmat(CMA.xmean,N,1) + CMA.sigma .* Step;
    Dec = RepairBoundsToMean(Dec,CMA.xmean,CMA.lower,CMA.upper);
    Step = CMAESStepsFromDec(CMA,Dec);
end
