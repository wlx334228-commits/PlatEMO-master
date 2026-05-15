function Theta = ActionToParam(action,Problem,Population,sigma0)

    PopDec = Population.decs;
    ulPop  = PopDec(:,1:Problem.DU);
    fitness = CalFitness(Problem.C,Population);

    [N,DU] = size(ulPop);

    if length(action) ~= 2*DU
        error('ActionToParam: action dimension mismatch. Expected %d, got %d.',2*DU,length(action));
    end

    [~,rank] = sort(fitness,'ascend');

    eliteNum = max(2,ceil(0.2*N));
    elitePop = ulPop(rank(1:eliteNum),:);

    eliteMean = mean(elitePop,1);

    popStd = std(ulPop,0,1);
    popStd(popStd < 1e-6) = 1e-6;

    minSigma = AdaptiveMinSigma(Problem,Population,DU);
    popStd = max(popStd,minSigma);

    rawDeltaMu    = action(1:DU);
    rawDeltaSigma = action(DU+1:2*DU);

    deltaMu = 0.25 * tanh(rawDeltaMu);
    deltaSigma = -0.3 + 0.8 * tanh(rawDeltaSigma);

    mu = eliteMean + deltaMu .* popStd;
    sigma = sigma0 .* exp(deltaSigma) .* popStd;

    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);

    mu = min(max(mu,lower),upper);
    sigma = max(sigma,1e-6);

    Theta.mu = mu;
    Theta.sigma = sigma;
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