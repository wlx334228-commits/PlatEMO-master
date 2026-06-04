function ulDec = GenerateUpperFromDistribution(Dist,N,Problem)
% Sample upper-level candidates from the actor-generated distribution.

    if N <= 0
        ulDec = zeros(0,Problem.DU);
        return;
    end

    ulDec = repmat(Dist.mu,N,1) + randn(N,Problem.DU) .* repmat(Dist.sigma,N,1);
    lowerU = repmat(Problem.lower(1:Problem.DU),N,1);
    upperU = repmat(Problem.upper(1:Problem.DU),N,1);
    ulDec = min(max(ulDec,lowerU),upperU);
end
