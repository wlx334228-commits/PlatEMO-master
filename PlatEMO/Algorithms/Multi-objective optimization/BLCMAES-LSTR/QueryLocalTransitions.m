function [direction,usable,info] = QueryLocalTransitions(xBase,Archive,Problem,params,mode)
% Query local transition knowledge around xBase.

    DU = Problem.DU;
    direction = zeros(1,DU);
    usable = false;
    info = EmptyQueryInfo(DU);

    if nargin < 5 || isempty(mode)
        mode = 'balanced';
    end

    [PositiveArchive,baseAlpha] = SelectPositiveArchive(Archive,Problem,params,mode);
    sourceX = ArchiveSourceX(PositiveArchive,Problem);
    if isempty(sourceX)
        return;
    end

    info.candidateAll = size(sourceX,1);
    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;

    dist = sqrt(sum(((sourceX - xBase) ./ range).^2,2)) / sqrt(DU);
    reuseThreshold = GetParam(params,'reuseThreshold',GetParam(params,'reuseRadius',inf));
    distanceIdx = find(dist <= reuseThreshold);
    info.candidateDistance = numel(distanceIdx);
    if isempty(distanceIdx)
        return;
    end

    improvement = ArchiveImprovement(PositiveArchive);
    minImprovement = GetParam(params,'minHistoricalImprovement',0);
    improvementIdx = distanceIdx(improvement(distanceIdx) >= minImprovement & isfinite(improvement(distanceIdx)));
    info.candidateImprovement = numel(improvementIdx);
    if isempty(improvementIdx)
        return;
    end

    Kuse = max(1,round(GetParam(params,'Kuse',GetParam(params,'neighborK',5))));
    selected = SelectKnowledgeByDistanceImprovement(dist(improvementIdx),improvement(improvementIdx),Kuse);
    local = improvementIdx(selected);
    info.candidateNDS = numel(local);
    info.usedN = numel(local);
    info.nNear = info.usedN;
    if info.usedN < max(1,round(GetParam(params,'minNear',1)))
        return;
    end

    localDist = dist(local);
    localD = PositiveArchive.D(local,:);
    localI = improvement(local);
    weights = 1 ./ (localDist + 1e-6);
    weights = weights ./ sum(weights);

    info.localDist = localDist;
    info.localD = localD;
    info.localI = localI;
    info.weights = weights;
    info.meanDist = mean(localDist);
    info.meanImprovement = mean(localI);

    direction = sum(localD .* repmat(weights,1,DU),1);
    directionNorms = sqrt(sum(localD.^2,2));
    info.consistency = norm(direction) / (sum(weights .* directionNorms) + eps);

    reliabilityEnabled = isfield(params,'enableIndividualReliability') && params.enableIndividualReliability;
    if ~reliabilityEnabled && info.consistency < params.minConsistency
        return;
    end

    if size(localD,1) >= 2 && params.directionNoise > 0
        centered = localD - repmat(direction,size(localD,1),1);
        localStd = sqrt(sum((centered.^2) .* repmat(weights,1,DU),1));
        direction = direction + params.directionNoise .* localStd .* randn(1,DU);
    end

    info.alpha = baseAlpha * min(1,info.consistency);
    if reliabilityEnabled
        usable = any(abs(direction) > 1e-12);
    else
        usable = info.alpha > 0 && any(abs(direction) > 1e-12);
    end
end

function info = EmptyQueryInfo(DU)
    info = struct('alpha',0,'nNear',0,'consistency',0, ...
        'candidateAll',0,'candidateDistance',0,'candidateImprovement',0, ...
        'candidateNDS',0,'usedN',0,'meanDist',0,'meanImprovement',0, ...
        'localDist',zeros(0,1),'localD',zeros(0,DU), ...
        'localI',zeros(0,1),'weights',zeros(0,1));
end

function selected = SelectKnowledgeByDistanceImprovement(dist,improvement,Kuse)
    n = numel(dist);
    if n <= Kuse
        selected = 1:n;
        return;
    end

    firstFront = true(n,1);
    for i = 1 : n
        dominated = (dist <= dist(i) & improvement >= improvement(i)) & ...
            (dist < dist(i) | improvement > improvement(i));
        dominated(i) = false;
        if any(dominated)
            firstFront(i) = false;
        end
    end

    selected = find(firstFront);
    if numel(selected) > Kuse
        [~,rank] = sort(dist(selected),'ascend');
        selected = selected(rank(1:Kuse));
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
            if isfield(Archive,'Balanced') && ~isempty(Archive.Balanced.X0)
                PositiveArchive = Archive.Balanced;
            else
                PositiveArchive = MergeTransitionSubArchives(Archive.Feasibility,Archive.Objective,Problem);
            end
            baseAlpha = params.alphaBalanced;
    end
end

function sourceX = ArchiveSourceX(SubArchive,Problem)
    if isfield(SubArchive,'XSource')
        sourceX = SubArchive.XSource;
    elseif isfield(SubArchive,'X0')
        sourceX = SubArchive.X0;
    else
        sourceX = zeros(0,Problem.DU);
    end
end

function improvement = ArchiveImprovement(SubArchive)
    if isfield(SubArchive,'I')
        improvement = SubArchive.I;
    elseif isfield(SubArchive,'W')
        improvement = SubArchive.W;
    else
        improvement = zeros(size(SubArchive.D,1),1);
    end
end

function SubArchive = MergeTransitionSubArchives(A,B,Problem)
    SubArchive.XSource = [ArchiveSourceX(A,Problem);ArchiveSourceX(B,Problem)];
    SubArchive.XTarget = [ArchiveTargetX(A,Problem);ArchiveTargetX(B,Problem)];
    SubArchive.X0 = SubArchive.XSource;
    SubArchive.D = [A.D;B.D];
    SubArchive.DPair = [ArchiveColumn(A,'DPair');ArchiveColumn(B,'DPair')];
    SubArchive.DeltaF = [ArchiveColumn(A,'DeltaF');ArchiveColumn(B,'DeltaF')];
    SubArchive.DeltaCV = [ArchiveColumn(A,'DeltaCV');ArchiveColumn(B,'DeltaCV')];
    SubArchive.I = [ArchiveImprovement(A);ArchiveImprovement(B)];
    SubArchive.W = SubArchive.I;
    SubArchive.Gen = [ArchiveColumn(A,'Gen');ArchiveColumn(B,'Gen')];
    SubArchive.MaxSize = A.MaxSize + B.MaxSize;
    if isempty(SubArchive.XSource)
        SubArchive = EmptyMergedArchive(Problem);
        SubArchive.MaxSize = A.MaxSize + B.MaxSize;
    end
end

function targetX = ArchiveTargetX(SubArchive,Problem)
    if isfield(SubArchive,'XTarget')
        targetX = SubArchive.XTarget;
    else
        targetX = ArchiveSourceX(SubArchive,Problem) + SubArchive.D;
    end
end

function values = ArchiveColumn(SubArchive,fieldName)
    if isfield(SubArchive,fieldName)
        values = SubArchive.(fieldName);
    else
        values = zeros(size(SubArchive.D,1),1);
    end
end

function SubArchive = EmptyMergedArchive(Problem)
    SubArchive.XSource = zeros(0,Problem.DU);
    SubArchive.XTarget = zeros(0,Problem.DU);
    SubArchive.X0 = zeros(0,Problem.DU);
    SubArchive.D = zeros(0,Problem.DU);
    SubArchive.DPair = zeros(0,1);
    SubArchive.DeltaF = zeros(0,1);
    SubArchive.DeltaCV = zeros(0,1);
    SubArchive.I = zeros(0,1);
    SubArchive.W = zeros(0,1);
    SubArchive.Gen = zeros(0,1);
    SubArchive.MaxSize = 0;
end

function value = GetParam(params,name,defaultValue)
    if isfield(params,name)
        value = params.(name);
    else
        value = defaultValue;
    end
end
