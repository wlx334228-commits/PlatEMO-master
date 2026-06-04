function stats = RewardCalculator(Problem,PPOOffspring,ppoSurvivalRate,ppoSurvived, ...
    oldBest,oldMean,eliteDist,modelDist)
% PPO reward uses only PPO distribution quality and PPO offspring behavior.

    if isempty(PPOOffspring)
        stats.reward = -1;
        stats.matchLoss = inf;
        stats.ppoMeanFit = inf;
        return;
    end

    ppoFit = CalFitness(Problem.C,PPOOffspring);
    ppoFit = ppoFit(:);

    muLoss = mean((modelDist.muNorm - eliteDist.muNorm).^2);
    sigmaLoss = mean((log(modelDist.sigmaNorm + 1e-12) - ...
        log(eliteDist.sigmaNorm + 1e-12)).^2);
    matchLoss = muLoss + 0.5 * sigmaLoss;
    matchReward = 2 * exp(-matchLoss) - 1;

    bestGain = (oldBest - min(ppoFit)) / (abs(oldBest) + 1e-8);
    meanGain = (oldMean - mean(ppoFit)) / (abs(oldMean) + 1e-8);
    survivalReward = 2 * ppoSurvivalRate - 1;

    qualityReward = 0.50 * survivalReward + ...
        0.30 * tanh(bestGain) + ...
        0.20 * tanh(meanGain);

    stats.reward = 0.70 * matchReward + 0.30 * qualityReward;
    stats.matchLoss = matchLoss;
    stats.muLoss = muLoss;
    stats.sigmaLoss = sigmaLoss;
    stats.matchReward = matchReward;
    stats.qualityReward = qualityReward;
    stats.ppoBestFit = min(ppoFit);
    stats.ppoMeanFit = mean(ppoFit);
    stats.ppoSurvivalRate = ppoSurvivalRate;
    stats.ppoSurvived = ppoSurvived;
end
