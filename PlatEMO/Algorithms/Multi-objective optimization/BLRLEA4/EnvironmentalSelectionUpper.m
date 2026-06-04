function [Population,ppoSurvivalRate,ppoSurvived] = EnvironmentalSelectionUpper( ...
    Problem,Population,Offspring,Nppo,ppoOffset)
% Upper-level environmental selection and PPO offspring survival tracking.

    N = length(Population);
    Pool = [Population,Offspring];
    [~,rank] = sort(CalFitness(Problem.C,Pool));
    selected = rank(1:N);
    Population = Pool(selected);

    if nargin < 4 || isempty(Nppo) || Nppo <= 0
        ppoSurvivalRate = 0;
        ppoSurvived = false(0,1);
        return;
    end
    if nargin < 5 || isempty(ppoOffset)
        ppoOffset = 1;
    end

    ppoStart = N + ppoOffset;
    ppoEnd = ppoStart + Nppo - 1;
    ppoSurvived = false(Nppo,1);
    survived = selected(selected >= ppoStart & selected <= ppoEnd) - ppoStart + 1;
    ppoSurvived(survived) = true;
    ppoSurvivalRate = mean(ppoSurvived);
end
