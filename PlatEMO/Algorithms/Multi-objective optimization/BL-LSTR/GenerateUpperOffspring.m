function offspring_ulPop = GenerateUpperOffspring(ul_Theta, Problem, N)
% GenerateUpperOffspring
% Generate upper-level offspring according to Gaussian distribution
%
% Input:
%   ul_Theta.mu       : 1 x DU mean vector
%   Theta.sigma    : 1 x DU std vector
%   Problem        : PlatEMO problem object
%   N              : number of offspring
%
% Output:
%   offspring_ulPop : N x DU upper-level offspring population

    DU = Problem.DU;

    % repeat distribution parameters
    MuMat    = repmat(ul_Theta.mu, N, 1);
    SigmaMat = repmat(ul_Theta.sigma, N, 1);

    % Gaussian sampling
    offspring_ulPop = MuMat + randn(N, DU) .* SigmaMat;

    % Boundary handling
    lower = repmat(Problem.lower(1:DU), N, 1);
    upper = repmat(Problem.upper(1:DU), N, 1);

    offspring_ulPop = min(max(offspring_ulPop, lower), upper);
end