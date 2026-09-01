function [ulOffDec,stats,genLog] = GenerateLSTROffspring(ulBaseDec,Archive,FeedbackArchive,Problem,params,CMA,mode)
% Soft-correct CMA-ES upper offspring by individual reliability gating.

    if isempty(ulBaseDec)
        ulOffDec = zeros(0,Problem.DU);
        stats = EmptyLSTRStats();
        genLog = EmptyGenerationLog(0,Problem.DU,'balanced');
        return;
    end

    if nargin < 7 || isempty(mode)
        mode = 'balanced';
    end

    DU = Problem.DU;
    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);
    range = upper - lower;
    range(range < 1e-12) = 1;

    ulOffDec = ulBaseDec;
    stats = EmptyLSTRStats();
    stats.total = size(ulBaseDec,1);
    genLog = EmptyGenerationLog(size(ulBaseDec,1),DU,mode);
    alphaSum = 0;
    for i = 1 : size(ulBaseDec,1)
        xBase = ulBaseDec(i,:);
        genLog(i).X = xBase;
        [dRaw,usable,info] = QueryLocalTransitions(xBase,Archive,Problem,params,mode);
        genLog(i).Usable = usable;
        genLog(i).NNear = info.nNear;
        genLog(i).CandidateAll = info.candidateAll;
        genLog(i).CandidateDistance = info.candidateDistance;
        genLog(i).CandidateImprovement = info.candidateImprovement;
        genLog(i).CandidateNDS = info.candidateNDS;
        genLog(i).UsedN = info.usedN;
        genLog(i).M = info.usedN;
        genLog(i).MeanDNew = info.meanDist;
        genLog(i).MeanDist = info.meanDist;
        genLog(i).MeanImprovement = info.meanImprovement;
        genLog(i).Consistency = info.consistency;
        genLog(i).DRawNorm = norm(dRaw);

        if usable
            stats.usable = stats.usable + 1;
            if params.enableIndividualReliability
                reliability = CalculateIndividualReliability(xBase,info,FeedbackArchive,Problem,params,mode);
                stepRaw = reliability.lambda .* dRaw;
                [step,sigmaClipped] = ApplySigmaSafetyCap(stepRaw,CMA,Problem,params);

                genLog(i).Rdist = reliability.Rdist;
                genLog(i).Rdir = reliability.Rdir;
                genLog(i).Rnum = reliability.Rnum;
                genLog(i).MeanDist = reliability.meanDist;
                genLog(i).M = reliability.M;
                genLog(i).R = reliability.R;
                genLog(i).Lambda = reliability.lambda;
                genLog(i).DeltaRawNorm = norm(stepRaw);
                genLog(i).DeltaFinalNorm = norm(step);
                genLog(i).SigmaClipped = sigmaClipped;
            else
                step = info.alpha .* dRaw;
                maxStep = params.maxStepRatio .* range;
                step = min(max(step,-maxStep),maxStep);

                if nargin >= 6 && ~isempty(CMA) && isfield(params,'stepSigmaRatio')
                    maxNorm = params.stepSigmaRatio * max(CMA.sigma,eps);
                    stepNorm = norm(step);
                    if stepNorm > maxNorm
                        step = step .* (maxNorm / stepNorm);
                    end
                end

                genLog(i).Lambda = info.alpha;
                genLog(i).DeltaRawNorm = norm(step);
                genLog(i).DeltaFinalNorm = norm(step);
            end

            if any(abs(step) > 1e-12)
                ulOffDec(i,:) = xBase + step;
                stats.accepted = stats.accepted + 1;
                genLog(i).Applied = true;
                if ~params.enableIndividualReliability
                    alphaSum = alphaSum + info.alpha;
                end
                if genLog(i).SigmaClipped
                    stats.sigmaClipped = stats.sigmaClipped + 1;
                end
            end
        end
    end

    if stats.accepted > 0 && ~params.enableIndividualReliability
        stats.meanAlpha = alphaSum / stats.accepted;
    end
    stats = SummarizeReliabilityStats(stats,genLog,params);

    ulOffDec = RepairBounds(ulOffDec,lower,upper);
end

function stats = EmptyLSTRStats()
    stats = struct('accepted',0,'total',0,'usable',0,'meanAlpha',0, ...
        'meanR',0,'medianR',0,'meanLambda',0,'zeroLambda',0,'sigmaClipped',0, ...
        'meanCandidateAll',0,'meanCandidateDistance',0,'meanCandidateImprovement',0, ...
        'meanCandidateNDS',0,'meanUsedN',0,'meanDNew',0,'meanImprovement',0, ...
        'meanM',0,'meanDist',0,'meanRdist',0,'meanRdir',0,'meanRnum',0, ...
        'meanDRawNorm',0,'meanLambdaDRawNorm',0,'meanDeltaFinalNorm',0, ...
        'noCorrectionRatio',0);
end

function genLog = EmptyGenerationLog(N,DU,mode)
    entry = struct('X',zeros(1,DU),'ModeCode',LSTRModeCode(mode), ...
        'Usable',false,'Applied',false,'NNear',0, ...
        'CandidateAll',0,'CandidateDistance',0,'CandidateImprovement',0, ...
        'CandidateNDS',0,'UsedN',0,'MeanDNew',0,'MeanImprovement',0, ...
        'M',0,'MeanDist',0,'Consistency',0, ...
        'DRawNorm',0,'Rdist',0,'Rdir',0,'Rnum',0,'R',0,'Lambda',0, ...
        'DeltaRawNorm',0,'DeltaFinalNorm',0,'SigmaClipped',false);
    genLog = repmat(entry,1,N);
end

function stats = SummarizeReliabilityStats(stats,genLog,params)
    if isempty(genLog)
        return;
    end

    total = max(1,numel(genLog));
    stats.meanCandidateAll = mean([genLog.CandidateAll]);
    stats.meanCandidateDistance = mean([genLog.CandidateDistance]);
    stats.meanCandidateImprovement = mean([genLog.CandidateImprovement]);
    stats.meanCandidateNDS = mean([genLog.CandidateNDS]);
    stats.meanUsedN = mean([genLog.UsedN]);
    stats.meanM = mean([genLog.M]);
    stats.meanDRawNorm = mean([genLog.DRawNorm]);
    stats.meanLambdaDRawNorm = mean([genLog.DeltaRawNorm]);
    stats.meanDeltaFinalNorm = mean([genLog.DeltaFinalNorm]);
    stats.noCorrectionRatio = 1 - stats.accepted / total;

    used = [genLog.UsedN] > 0;
    if any(used)
        stats.meanDNew = mean([genLog(used).MeanDNew]);
        stats.meanDist = mean([genLog(used).MeanDist]);
        stats.meanImprovement = mean([genLog(used).MeanImprovement]);
    end

    if ~params.enableIndividualReliability
        return;
    end

    usable = [genLog.Usable];
    if ~any(usable)
        return;
    end

    R = [genLog(usable).R];
    Lambda = [genLog(usable).Lambda];
    stats.meanRdist = mean([genLog(usable).Rdist]);
    stats.meanRdir = mean([genLog(usable).Rdir]);
    stats.meanRnum = mean([genLog(usable).Rnum]);
    stats.meanR = mean(R);
    stats.medianR = median(R);
    stats.meanLambda = mean(Lambda);
    stats.zeroLambda = sum(Lambda <= 0);
end
