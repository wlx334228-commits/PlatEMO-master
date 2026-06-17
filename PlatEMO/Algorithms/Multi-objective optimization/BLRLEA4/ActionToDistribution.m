function Dist = ActionToDistribution(action,Problem)
% Convert actor output [muNorm,sigmaNorm] into an upper-level Gaussian distribution.

    DU = Problem.DU;
    if length(action) ~= 2 * DU
        error('ActionToDistribution: action dimension mismatch.');
    end

    lowerU = Problem.lower(1:DU);
    upperU = Problem.upper(1:DU);
    rangeU = upperU - lowerU;
    rangeU(rangeU < 1e-12) = 1;

    muNorm = action(1:DU);
    sigmaMinNorm = 1e-5;
    sigmaMaxNorm = 0.50;
    sigmaNorm = action(DU+1:2*DU);

    Dist.muNorm = min(max(muNorm,0),1);
    Dist.sigmaNorm = min(max(sigmaNorm,sigmaMinNorm),sigmaMaxNorm);
    Dist.mu = lowerU + Dist.muNorm .* rangeU;
    Dist.sigma = Dist.sigmaNorm .* rangeU;
end
