function [rewards,stats] = RewardCalculator( ...
    Problem,Offspring,Nbase,Nppo,deltaPPO,perturbScale)
% Paired reward for PPO perturbations.
%
% Offspring(1:Nbase) are elite-distribution anchors without perturbation.
% Offspring(Nbase+1:Nbase+Nppo) are the same anchors with PPO perturbation.
% EA offspring are ignored here.

    pairN = min(Nbase,Nppo);
    BaseOffspring = Offspring(1:pairN);
    PPOOffspring  = Offspring(Nbase+1:Nbase+pairN);

    baseFit = CalFitness(Problem.C,BaseOffspring);
    ppoFit  = CalFitness(Problem.C,PPOOffspring);
    baseFit = baseFit(:);
    ppoFit  = ppoFit(:);

    if nargin < 5 || isempty(deltaPPO)
        deltaPPO = zeros(pairN,Problem.DU);
    else
        deltaPPO = deltaPPO(1:pairN,:);
    end
    if nargin < 6 || isempty(perturbScale)
        perturbScale = ones(1,Problem.DU);
    end

    if isvector(perturbScale)
        perturbScale = perturbScale(:)';
        scaleMat = repmat(perturbScale,pairN,1);
    else
        scaleMat = perturbScale(1:pairN,:);
    end
    stepSize = mean(abs(deltaPPO ./ (scaleMat + 1e-12)),2);

    relImprove = (baseFit - ppoFit) ./ (abs(baseFit) + 1e-8);
    rImprove   = tanh(relImprove);
    rBetter    = 2 * double(ppoFit < baseFit) - 1;
    rStep      = -stepSize;

    rewards = 0.75 * rImprove + 0.20 * rBetter + 0.05 * rStep;
    rewards = max(min(rewards,1),-1);

    stats.betterRate = mean(ppoFit < baseFit);
    stats.meanBaseFit = mean(baseFit);
    stats.meanPPOFit  = mean(ppoFit);
    stats.meanImprove = mean(relImprove);
    stats.meanReward  = mean(rewards);
    stats.meanStep    = mean(stepSize);
end
