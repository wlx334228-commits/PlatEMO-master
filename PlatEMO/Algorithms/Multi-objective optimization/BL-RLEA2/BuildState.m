function state = BuildState(Problem,Population,oldBest,noImproveGen, ...
    prevPPOBetterRate,prevPPOSurvivalRate,prevMeanSigma)

    PopDec = Population.decs;
    PopCon = Population.cons;

    ulPop = PopDec(:,1:Problem.DU);
    ulFit = CalFitness(Problem.C,Population);
    N = size(ulPop,1);

    if isempty(PopCon) || Problem.C <= 0
        ulCon = zeros(N,1);
    else
        ulCon = PopCon(:,1:Problem.C);
    end

    bestFit = min(ulFit);

    [~,rank] = sort(ulFit,'ascend');
    eliteNum = max(2,ceil(0.2*numel(rank)));
    eliteIdx = rank(1:eliteNum);
    elitePop = ulPop(eliteIdx,:);
    eliteMeanFit = mean(ulFit(eliteIdx));

    rangeU = Problem.upper(1:Problem.DU) - Problem.lower(1:Problem.DU);
    rangeU(rangeU < 1e-12) = 1;

    if size(ulPop,1) == 1
        popStd = zeros(1,Problem.DU);
    else
        popStd = std(ulPop,0,1);
    end

    if size(elitePop,1) == 1
        eliteStd = zeros(1,Problem.DU);
    else
        eliteStd = std(elitePop,0,1);
    end

    feasULRate = mean(all(ulCon <= 0,2));

    s1 = signedLog(bestFit);
    s2 = signedLog(eliteMeanFit);
    s3 = tanh((oldBest - bestFit) / (abs(oldBest) + 1e-8));
    s4 = tanh(noImproveGen / 5);
    s5 = log1p(mean(popStd ./ (rangeU + 1e-12)));
    s6 = log1p(mean(eliteStd ./ (rangeU + 1e-12)));
    s7 = 2 * feasULRate - 1;
    s8 = 2 * prevPPOBetterRate - 1;
    s9 = 2 * prevPPOSurvivalRate - 1;
    s10 = signedLog(prevMeanSigma / (mean(rangeU) + 1e-12));

    state = [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10];
end

function y = signedLog(x)
    y = sign(x).*log1p(abs(x));
end
