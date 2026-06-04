function state = BuildPerturbState(Problem,baseUL,eliteInfo)
% Build per-dimension PPO state for the current perturbation anchor.

    DU = Problem.DU;

    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);
    range = upper - lower;
    range(range < 1e-12) = 1;

    xNorm = 2 .* (baseUL - lower) ./ range - 1;
    eliteStdNorm = eliteInfo.perturbScale ./ range;
    eliteStdNorm = min(max(eliteStdNorm,0),1);

    state = [xNorm,eliteStdNorm];
end
