function [state,eliteDist] = BuildState(Problem,Population,~,~)
% Build elite-distribution-only state and the supervised target.

    PopDec = Population.decs;
    ulPop = PopDec(:,1:Problem.DU);
    fit = CalFitness(Problem.C,Population);
    N = size(ulPop,1);

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

    state = [eliteDist.muNorm,eliteDist.sigmaNorm];
    state(~isfinite(state)) = 0;
end
