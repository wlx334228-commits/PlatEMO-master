function CMA = InitializeCMAES(D,lower,upper,lambda,xmean)
% Initialize one CMA-ES state for a bounded real search space.

    lower = lower(:)';
    upper = upper(:)';
    range = upper - lower;
    range(range < 1e-12) = 1;

    lambda = max(4,lambda);
    mu = max(1,round(lambda/2));
    weights = log(mu+0.5) - log(1:mu);
    weights = weights ./ sum(weights);
    mueff = 1 / sum(weights.^2);

    if nargin < 5 || isempty(xmean)
        xmean = unifrnd(lower,upper);
    else
        xmean = min(max(xmean(:)',lower),upper);
    end

    CMA.D = D;
    CMA.lambda = lambda;
    CMA.mu = mu;
    CMA.weights = weights;
    CMA.mueff = mueff;
    CMA.cs = (mueff+2)/(D+mueff+5);
    CMA.ds = 1 + CMA.cs + 2*max(sqrt((mueff-1)/(D+1))-1,0);
    CMA.cc = (4+mueff/D)/(4+D+2*mueff/D);
    CMA.c1 = 2/((D+1.3)^2+mueff);
    CMA.cmu = min(1-CMA.c1,2*(mueff-2+1/mueff)/((D+2)^2+2*mueff/2));
    CMA.ENN = sqrt(D)*(1-1/(4*D)+1/(21*D^2));
    CMA.hth = (1.4+2/(D+1))*CMA.ENN;
    CMA.lower = lower;
    CMA.upper = upper;
    CMA.range = range;
    CMA.xmean = xmean;
    CMA.sigma = 0.1 .* range;
    CMA.minSigma = max(1e-8 .* range,1e-12);
    CMA.maxSigma = max(0.5 .* range,CMA.minSigma);
    CMA.ps = zeros(1,D);
    CMA.pc = zeros(1,D);
    CMA.C = eye(D);
    CMA.iter = 0;
end
