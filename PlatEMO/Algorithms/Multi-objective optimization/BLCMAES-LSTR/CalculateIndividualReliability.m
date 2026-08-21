function reliability = CalculateIndividualReliability(xBase,neighborInfo,FeedbackArchive,Problem,params,mode)
% Estimate how much of the raw LSTR correction should be retained.

    reliability.Rdist = DistanceReliability(neighborInfo,params);
    reliability.Rdir = DirectionReliability(neighborInfo);
    reliability.Rhist = HistoricalFeedbackReliability(xBase,FeedbackArchive,Problem,params,mode);

    w = max(0,[params.wd,params.wc,params.wh]);
    if sum(w) <= 0
        w = [0.30,0.35,0.35];
    end
    w = w ./ sum(w);

    reliability.R = w(1)*reliability.Rdist + ...
        w(2)*reliability.Rdir + w(3)*reliability.Rhist;
    reliability.R = min(max(reliability.R,0),1);
    reliability.lambda = ReliabilityGate(reliability.R,params);
end

function Rdist = DistanceReliability(neighborInfo,params)
    dist = neighborInfo.localDist(:);
    if isempty(dist)
        Rdist = 0;
        return;
    end

    D = mean(dist);
    if isfield(params,'betaReliability') && params.betaReliability > 0
        beta = params.betaReliability;
    else
        beta = max([median(dist),params.reuseThreshold/2,eps]);
    end
    Rdist = exp(-D / beta);
    Rdist = min(max(Rdist,0),1);
end

function Rdir = DirectionReliability(neighborInfo)
    localD = neighborInfo.localD;
    if isempty(localD)
        Rdir = 0.5;
        return;
    end

    directionNorms = sqrt(sum(localD.^2,2));
    valid = directionNorms > 1e-12;
    localD = localD(valid,:);
    directionNorms = directionNorms(valid);
    M = size(localD,1);
    if M < 2
        Rdir = 0.5;
        return;
    end

    unitD = localD ./ repmat(directionNorms,1,size(localD,2));
    cosSum = 0;
    pairN = 0;
    for i = 1 : M-1
        for j = i+1 : M
            cosSum = cosSum + unitD(i,:) * unitD(j,:)';
            pairN = pairN + 1;
        end
    end

    C = cosSum / max(pairN,1);
    Rdir = (C + 1) / 2;
    Rdir = min(max(Rdir,0),1);
end

function Rhist = HistoricalFeedbackReliability(xBase,FeedbackArchive,Problem,params,mode)
    Rhist = 0.5;
    if isempty(FeedbackArchive) || isempty(FeedbackArchive.X)
        return;
    end

    modeCode = LSTRModeCode(mode);
    sameMode = FeedbackArchive.ModeCode == modeCode;
    applied = FeedbackArchive.Applied;
    candidate = sameMode & applied;
    if ~any(candidate)
        return;
    end

    DU = Problem.DU;
    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;
    X = FeedbackArchive.X(candidate,:);
    dist = sqrt(sum(((X - xBase) ./ range).^2,2));
    valid = dist <= params.reuseThreshold;
    if ~any(valid)
        return;
    end

    candidateIdx = find(candidate);
    candidateIdx = candidateIdx(valid);
    dist = dist(valid);
    [~,rank] = sort(dist,'ascend');
    K = min(max(1,round(params.feedbackK)),numel(rank));
    nearIdx = candidateIdx(rank(1:K));

    S = sum(FeedbackArchive.Success(nearIdx));
    Nh = numel(nearIdx);
    a = max(params.feedbackPriorA,0);
    b = max(params.feedbackPriorB,0);
    Rhist = (S + a) / (Nh + a + b);
    Rhist = min(max(Rhist,0),1);
end

function lambda = ReliabilityGate(R,params)
    tau = min(max(params.tauR,0),1);
    if R <= tau || tau >= 1
        lambda = 0;
    else
        lambda = (R - tau) / (1 - tau);
    end
    lambda = min(max(lambda,0),1);
end
