function [ulOffDec,stats] = GenerateLSTROffspring(ulBaseDec,Archive,Problem,params,CMA,mode)
% Soft-correct CMA-ES upper offspring by reliable local transition reuse.

    if isempty(ulBaseDec)
        ulOffDec = zeros(0,Problem.DU);
        stats = struct('accepted',0,'total',0,'meanAlpha',0);
        return;
    end

    if nargin < 6 || isempty(mode)
        mode = 'balanced';
    end

    DU = Problem.DU;
    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);
    range = upper - lower;
    range(range < 1e-12) = 1;

    ulOffDec = ulBaseDec;
    stats = struct('accepted',0,'total',size(ulBaseDec,1),'meanAlpha',0);
    alphaSum = 0;
    for i = 1 : size(ulBaseDec,1)
        [direction,usable,info] = QueryLocalTransitions(ulBaseDec(i,:),Archive,Problem,params,mode);
        if usable
            step = info.alpha .* direction;
            maxStep = params.maxStepRatio .* range;
            step = min(max(step,-maxStep),maxStep);

            if nargin >= 5 && ~isempty(CMA) && isfield(params,'stepSigmaRatio')
                maxNorm = params.stepSigmaRatio * max(CMA.sigma,eps);
                stepNorm = norm(step);
                if stepNorm > maxNorm
                    step = step .* (maxNorm / stepNorm);
                end
            end

            if any(abs(step) > 1e-12)
                ulOffDec(i,:) = ulBaseDec(i,:) + step;
                stats.accepted = stats.accepted + 1;
                alphaSum = alphaSum + info.alpha;
            end
        end
    end

    if stats.accepted > 0
        stats.meanAlpha = alphaSum / stats.accepted;
    end

    ulOffDec = RepairBounds(ulOffDec,lower,upper);
end
