function [direction,usable] = QueryLocalTransitions(xBase,Archive,Problem,params)
% Query locally relevant successful transitions around xBase.

    DU = Problem.DU;
    direction = zeros(1,DU);
    usable = false;

    if isempty(Archive.X0)
        return;
    end

    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;

    dist = sqrt(sum(((Archive.X0 - xBase) ./ range).^2,2));
    [sortedDist,rank] = sort(dist,'ascend');

    K = min(params.neighborK,numel(rank));
    local = rank(1:K);
    localDist = sortedDist(1:K);

    valid = localDist <= params.reuseThreshold;
    if ~any(valid)
        return;
    end

    local = local(valid);
    localDist = localDist(valid);
    localD = Archive.D(local,:);
    localW = Archive.W(local);
    localW = max(localW,1e-12);

    weights = localW ./ (localDist + 1e-6);
    weights = weights ./ sum(weights);

    meanD = sum(localD .* repmat(weights,1,DU),1);

    if size(localD,1) >= 2 && params.directionNoise > 0
        centered = localD - repmat(meanD,size(localD,1),1);
        localStd = sqrt(sum((centered.^2) .* repmat(weights,1,DU),1));
        direction = meanD + params.directionNoise .* localStd .* randn(1,DU);
    else
        direction = meanD;
    end

    usable = any(abs(direction) > 1e-12);
end
