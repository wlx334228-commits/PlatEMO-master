function Dist = ActionToDistribution(action,Problem)
% Convert actor action directly into upper-level Gaussian distribution.

    DU = Problem.DU;
    if length(action) ~= 2 * DU
        error('ActionToDistribution: action dimension mismatch.');
    end

    lowerU = Problem.lower(1:DU);
    upperU = Problem.upper(1:DU);
    rangeU = upperU - lowerU;
    rangeU(rangeU < 1e-12) = 1;

    rawMu = action(1:DU);
    rawSigma = action(DU+1:2*DU);

    muNorm = sigmoid(rawMu);
    sigmaMinNorm = 1e-5;
    sigmaMaxNorm = 0.50;
    sigmaNorm = sigmaMinNorm + sigmoid(rawSigma) .* (sigmaMaxNorm - sigmaMinNorm);

    Dist.muNorm = min(max(muNorm,0),1);
    Dist.sigmaNorm = min(max(sigmaNorm,sigmaMinNorm),sigmaMaxNorm);
    Dist.mu = lowerU + Dist.muNorm .* rangeU;
    Dist.sigma = Dist.sigmaNorm .* rangeU;
end

function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
end
