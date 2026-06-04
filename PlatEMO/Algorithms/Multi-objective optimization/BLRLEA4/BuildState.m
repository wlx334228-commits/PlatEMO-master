function [state,eliteDist] = BuildState(Problem,Population,oldBest,noImproveGen)
% Build population state and the supervised elite-distribution target.

    PopDec = Population.decs;
    PopCon = Population.cons;
    ulPop = PopDec(:,1:Problem.DU);
    fit = CalFitness(Problem.C,Population);
    N = size(ulPop,1);

    if isempty(PopCon) || Problem.C <= 0
        ulCon = zeros(N,1);
    else
        ulCon = PopCon(:,1:Problem.C);
    end

    lowerU = Problem.lower(1:Problem.DU);
    upperU = Problem.upper(1:Problem.DU);
    rangeU = upperU - lowerU;
    rangeU(rangeU < 1e-12) = 1;

    [~,rank] = sort(fit,'ascend');
    eliteNum = max(2,ceil(0.2*N));
    elitePop = ulPop(rank(1:eliteNum),:);

    eliteMu = mean(elitePop,1);
    if size(elitePop,1) == 1
        eliteSigma = 1e-5 .* rangeU;
    else
        eliteSigma = std(elitePop,0,1);
    end
    eliteSigma = max(eliteSigma,1e-5 .* rangeU);
    eliteSigma = min(eliteSigma,0.50 .* rangeU);

    eliteDist.mu = eliteMu;
    eliteDist.sigma = eliteSigma;
    eliteDist.muNorm = min(max((eliteMu - lowerU) ./ rangeU,0),1);
    eliteDist.sigmaNorm = min(max(eliteSigma ./ rangeU,1e-5),0.50);

    bestFit = min(fit);
    meanFit = mean(fit);
    stdFit = std(fit);
    if N <= 1
        diversity = 0;
    else
        diversity = mean(std(ulPop ./ rangeU,0,1));
    end
    feasibleRate = mean(all(ulCon <= 0,2));
    improvement = (oldBest - bestFit) / (abs(oldBest) + 1e-8);
    fitScale = abs(meanFit) + 1e-8;
    genProgress = min(Problem.FE / max(Problem.maxFE,1),1);

    globalState = [ ...
        signedLog(bestFit), ...
        signedLog(meanFit), ...
        tanh(stdFit / fitScale), ...
        tanh(diversity), ...
        tanh(improvement), ...
        tanh(noImproveGen / 5), ...
        2 * feasibleRate - 1, ...
        2 * genProgress - 1];

    eliteMuState = 2 * eliteDist.muNorm - 1;
    eliteSigmaState = log1p(eliteDist.sigmaNorm);

    state = [globalState,eliteMuState,eliteSigmaState];
    state(~isfinite(state)) = 0;
end

function y = signedLog(x)
    y = sign(x).*log1p(abs(x));
end
