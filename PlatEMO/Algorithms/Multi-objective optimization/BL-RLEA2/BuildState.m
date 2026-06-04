function state = BuildState(Problem,Population,oldBest,noImproveGen)

    PopDec = Population.decs;
    ulPop = PopDec(:,1:Problem.DU);
    ulFit = CalFitness(Problem.C,Population);

    bestFit = min(ulFit);

    [~,rank] = sort(ulFit,'ascend');
    eliteNum = max(2,ceil(0.2*numel(rank)));
    eliteIdx = rank(1:eliteNum);
    elitePop = ulPop(eliteIdx,:);

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

    populationDiversity = mean(popStd ./ (rangeU + 1e-12));
    eliteDiversity = mean(eliteStd ./ (rangeU + 1e-12));
    diversityRatio = eliteDiversity / (populationDiversity + 1e-12);

    s1 = signedLog(bestFit);
    s2 = log1p(populationDiversity);
    s3 = log1p(eliteDiversity);
    s4 = log1p(diversityRatio);
    s5 = tanh((oldBest - bestFit) / (abs(oldBest) + 1e-8));
    s6 = tanh(noImproveGen / 5);

    state = [s1,s2,s3,s4,s5,s6];
end

function y = signedLog(x)
    y = sign(x).*log1p(abs(x));
end
