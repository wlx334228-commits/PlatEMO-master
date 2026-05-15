function state = BuildPerturbState(Problem,globalState,baseUL,eliteInfo)
% Add candidate-level features so PPO can perturb each anchor differently.

    DU = Problem.DU;

    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);
    range = upper - lower;
    range(range < 1e-12) = 1;

    xNorm = 2 .* (baseUL - lower) ./ range - 1;
    zElite = (baseUL - eliteInfo.eliteMean) ./ (eliteInfo.perturbScale + 1e-12);

    distElite = norm(zElite) / sqrt(DU);
    meanScale = mean(eliteInfo.perturbScale ./ range);

    state = [globalState,xNorm,zElite,distElite,meanScale];
end
