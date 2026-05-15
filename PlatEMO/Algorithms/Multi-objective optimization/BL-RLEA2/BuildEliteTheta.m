function [Theta,eliteInfo] = BuildEliteTheta(Problem,Population,sigma0)
% Build the elite-sampling distribution used to generate PPO anchor points.

    PopDec  = Population.decs;
    ulPop   = PopDec(:,1:Problem.DU);
    fitness = CalFitness(Problem.C,Population);

    [N,DU] = size(ulPop);
    [~,rank] = sort(fitness,'ascend');

    eliteNum = max(2,ceil(0.2*N));
    elitePop = ulPop(rank(1:eliteNum),:);
    eliteMean = mean(elitePop,1);

    popStd = std(ulPop,0,1);
    popStd(popStd < 1e-6) = 1e-6;

    minSigma = AdaptiveMinSigma(Problem,Population,DU);
    perturbScale = max(popStd,minSigma);

    Theta.mu = eliteMean;
    Theta.sigma = sigma0 .* exp(-0.3) .* perturbScale;

    eliteInfo.eliteMean = eliteMean;
    eliteInfo.elitePop = elitePop;
    eliteInfo.popStd = popStd;
    eliteInfo.perturbScale = perturbScale;
end

function minSigma = AdaptiveMinSigma(Problem,Population,DU)

    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;

    fitness = CalFitness(Problem.C,Population);
    bestAbs = abs(min(fitness));

    if bestAbs > 1e-2
        scale = 0.02;
    elseif bestAbs > 1e-4
        scale = 0.005;
    else
        scale = 1e-5;
    end

    minSigma = max(scale .* range,1e-6);
end
