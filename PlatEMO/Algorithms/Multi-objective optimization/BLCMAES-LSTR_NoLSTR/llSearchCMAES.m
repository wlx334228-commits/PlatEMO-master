function [eliteIndiv,totalFElower] = llSearchCMAES(Problem,ulPopDec,globalCMA,totalFElower)
% Obtain the lower-level response by CMA-ES from the global lower marginal.

    lower = Problem.lower(Problem.DU+1:end);
    upper = Problem.upper(Problem.DU+1:end);
    lowerTol = 1e-6;
    sigma0 = 1;
    lowerCMA = InitializeLowerCMAESFromGlobal(Problem,globalCMA);

    bestIndiv = [];
    maxIter = ceil(Problem.maxFElower / lowerCMA.lambda);
    imprIter = max(2,ceil(0.2 * maxIter));
    record = nan(1,maxIter);

    FElower = 0;
    for iter = 1 : maxIter
        batchSize = min(lowerCMA.lambda,Problem.maxFElower-FElower);
        if batchSize <= 0
            break;
        end

        [llOffDec,~] = SampleCMAES(lowerCMA,batchSize);
        llOffspring = Problem.EvaluationLower([repmat(ulPopDec,batchSize,1),llOffDec]);

        FElower = FElower + length(llOffspring);
        totalFElower = totalFElower + length(llOffspring);

        lowerCMA = UpdateCMAES(lowerCMA,llOffDec,CalFitness(Problem.C,llOffspring));

        bestIndiv = UpdateBestLowerIndiv(Problem,bestIndiv,llOffspring);
        record(iter) = CalLowerFitnessValue(Problem,bestIndiv);

        reachFlatRatio = false;
        reachFlatValue = false;
        if iter > imprIter
            oldFit = record(iter-imprIter+1);
            curFit = record(iter);
            denom = abs(record(1)) + abs(curFit) + eps;
            reachFlatRatio = abs(curFit-oldFit) / denom < 1e-4;
            reachFlatValue = abs(curFit-oldFit) < 10 * lowerTol;
        end

        if (reachFlatRatio && reachFlatValue) || ...
                lowerCMA.sigma / sigma0 < 1e-2 || ...
                lowerCMA.sigma / sigma0 > 1e2
            break;
        end
    end

    if isempty(bestIndiv)
        eliteIndiv = lowerCMA.xmean;
    else
        eliteIndiv = bestIndiv.dec(Problem.DU+1:end);
    end
end

function fit = CalLowerFitnessValue(Problem,Population)
    fit = min(CalFitness(Problem.C,Population));
end

function lowerCMA = InitializeLowerCMAESFromGlobal(Problem,globalCMA)
    lower = Problem.lower(Problem.DU+1:end);
    upper = Problem.upper(Problem.DU+1:end);
    lambda = 4 + floor(3*log(max(Problem.DL,1)));
    xmean = globalCMA.xmean(Problem.DU+1:end);

    lowerCMA = InitializeCMAES(Problem.DL,lower,upper,lambda,xmean);
    lowerBlock = Problem.DU+1 : Problem.D;
    lowerCMA.sigma = 1;
    lowerCMA.minSigma = 1e-10;
    lowerCMA.maxSigma = 1e2;
    lowerCMA.C = globalCMA.C(lowerBlock,lowerBlock) * globalCMA.sigma^2;
    lowerCMA.pc = globalCMA.pc(lowerBlock) * globalCMA.sigma;
    lowerCMA.ps = zeros(1,Problem.DL);
    lowerCMA = RepairCMAES(lowerCMA);
end

function bestIndiv = UpdateBestLowerIndiv(Problem,bestIndiv,llOffspring)
    if isempty(bestIndiv)
        pool = llOffspring;
    else
        pool = [bestIndiv,llOffspring];
    end
    [~,best] = min(CalFitness(Problem.C,pool));
    bestIndiv = pool(best);
end
