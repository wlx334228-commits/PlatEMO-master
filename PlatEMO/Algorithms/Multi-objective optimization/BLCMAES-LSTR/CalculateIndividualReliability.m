function reliability = CalculateIndividualReliability(~,neighborInfo,~,~,params,~)
% Estimate how much of the raw LSTR correction should be retained.

    reliability.M = SupportCount(neighborInfo);
    [reliability.Rdist,reliability.meanDist] = DistanceReliability(neighborInfo,params);
    reliability.Rdir = DirectionReliability(neighborInfo);
    reliability.Rnum = NumberReliability(reliability.M,params);

    reliability.R = reliability.Rdist * reliability.Rdir * reliability.Rnum;
    reliability.R = min(max(reliability.R,0),1);
    reliability.lambda = ReliabilityGate(reliability.R,params);
end

function M = SupportCount(neighborInfo)
    if isfield(neighborInfo,'usedN')
        M = neighborInfo.usedN;
    else
        M = size(neighborInfo.localD,1);
    end
end

function [Rdist,meanDist] = DistanceReliability(neighborInfo,params)
    dist = neighborInfo.localDist(:);
    if isempty(dist)
        meanDist = inf;
        Rdist = 0;
        return;
    end

    reuseThreshold = GetParam(params,'reuseThreshold',GetParam(params,'reuseRadius',inf));
    meanDist = mean(dist);
    if reuseThreshold <= 0
        Rdist = 0;
    else
        Rdist = 1 - meanDist / reuseThreshold;
    end
    Rdist = min(max(Rdist,0),1);
end

function Rdir = DirectionReliability(neighborInfo)
    localD = neighborInfo.localD;
    if isempty(localD)
        Rdir = 0;
        return;
    end

    weights = neighborInfo.weights(:);
    if numel(weights) ~= size(localD,1) || sum(weights) <= 0
        weights = ones(size(localD,1),1) ./ size(localD,1);
    else
        weights = weights ./ sum(weights);
    end

    directionNorms = sqrt(sum(localD.^2,2));
    valid = directionNorms > 1e-12 & isfinite(directionNorms);
    if ~any(valid)
        Rdir = 0;
        return;
    end

    localD = localD(valid,:);
    directionNorms = directionNorms(valid);
    weights = weights(valid);
    if sum(weights) <= 0
        weights = ones(numel(weights),1) ./ numel(weights);
    else
        weights = weights ./ sum(weights);
    end

    unitD = localD ./ repmat(directionNorms + 1e-12,1,size(localD,2));
    uConsensus = sum(unitD .* repmat(weights,1,size(unitD,2)),1);
    Rdir = norm(uConsensus);
    Rdir = min(max(Rdir,0),1);
end

function Rnum = NumberReliability(M,params)
    Mref = max(1,round(GetParam(params,'Mref',3)));
    Rnum = min(1,M / Mref);
    Rnum = min(max(Rnum,0),1);
end

function value = GetParam(params,name,defaultValue)
    if isfield(params,name)
        value = params.(name);
    else
        value = defaultValue;
    end
end

function lambda = ReliabilityGate(R,params)
    if isfield(params,'reliabilityThreshold')
        tau = params.reliabilityThreshold;
    else
        tau = params.tauR;
    end
    tau = min(max(tau,0),1);
    if R <= tau || tau >= 1
        lambda = 0;
    else
        lambda = (R - tau) / (1 - tau);
    end
    lambda = min(max(lambda,0),1);
end
