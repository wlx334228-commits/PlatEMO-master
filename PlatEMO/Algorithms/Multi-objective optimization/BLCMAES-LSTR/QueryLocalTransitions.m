function [direction,usable,info] = QueryLocalTransitions(xBase,Archive,Problem,params,mode)
% Query a raw constraint-aware local transition around xBase.

    DU = Problem.DU;
    direction = zeros(1,DU);
    usable = false;
    info = struct('alpha',0,'nNear',0,'consistency',0, ...
        'localDist',zeros(0,1),'localD',zeros(0,DU),'weights',zeros(0,1));

    if nargin < 5 || isempty(mode)
        mode = 'balanced';
    end

    [PositiveArchive,baseAlpha] = SelectPositiveArchive(Archive,Problem,params,mode);
    if isempty(PositiveArchive.X0)
        return;
    end

    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;

    dist = sqrt(sum(((PositiveArchive.X0 - xBase) ./ range).^2,2));
    [sortedDist,rank] = sort(dist,'ascend');

    K = min(params.neighborK,numel(rank));
    local = rank(1:K);
    localDist = sortedDist(1:K);

    valid = localDist <= params.reuseThreshold;
    local = local(valid);
    localDist = localDist(valid);
    info.nNear = numel(local);
    if info.nNear < params.minNear
        return;
    end

    localD = PositiveArchive.D(local,:);
    localW = max(PositiveArchive.W(local),1e-12);
    weights = localW ./ (localDist + 1e-6);
    weights = weights ./ sum(weights);
    info.localDist = localDist;
    info.localD = localD;
    info.weights = weights;

    meanD = sum(localD .* repmat(weights,1,DU),1);
    directionNorms = sqrt(sum(localD.^2,2));
    info.consistency = norm(meanD) / (sum(weights .* directionNorms) + eps);
    reliabilityEnabled = isfield(params,'enableIndividualReliability') && params.enableIndividualReliability;
    if ~reliabilityEnabled && info.consistency < params.minConsistency
        return;
    end

    if size(localD,1) >= 2 && params.directionNoise > 0
        centered = localD - repmat(meanD,size(localD,1),1);
        localStd = sqrt(sum((centered.^2) .* repmat(weights,1,DU),1));
        direction = meanD + params.directionNoise .* localStd .* randn(1,DU);
    else
        direction = meanD;
    end

    info.alpha = baseAlpha * min(1,info.consistency);
    if reliabilityEnabled
        usable = any(abs(direction) > 1e-12);
    else
        usable = info.alpha > 0 && any(abs(direction) > 1e-12);
    end
end

function [PositiveArchive,baseAlpha] = SelectPositiveArchive(Archive,Problem,params,mode)
    switch mode
        case 'feasibility'
            PositiveArchive = Archive.Feasibility;
            baseAlpha = params.alphaFeasibility;
        case 'objective'
            PositiveArchive = Archive.Objective;
            baseAlpha = params.alphaObjective;
        otherwise
            PositiveArchive = MergeTransitionSubArchives(Archive.Feasibility,Archive.Objective,Problem);
            baseAlpha = params.alphaBalanced;
    end
end

function SubArchive = MergeTransitionSubArchives(A,B,Problem)
    SubArchive.X0 = [A.X0;B.X0];
    SubArchive.D = [A.D;B.D];
    SubArchive.W = [A.W;B.W];
    SubArchive.Gen = [A.Gen;B.Gen];
    SubArchive.MaxSize = A.MaxSize + B.MaxSize;
    if isempty(SubArchive.X0)
        SubArchive.X0 = zeros(0,Problem.DU);
        SubArchive.D = zeros(0,Problem.DU);
        SubArchive.W = zeros(0,1);
        SubArchive.Gen = zeros(0,1);
    end
end
