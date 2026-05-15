function [Population,ppoSurvivalRate,ppoSurvived] = EnvironmentalSelectionUpper(Problem,Population,Offspring,Nppo)
% Generational environmental selection for upper-level population

    N = length(Population);
    Pool = [Population,Offspring];

    [~,rank] = sort(CalFitness(Problem.C,Pool));

    selected = rank(1:N);
    Population = Pool(selected);

    if nargin < 4 || isempty(Nppo) || Nppo <= 0
        ppoSurvivalRate = 0;
        ppoSurvived = false(0,1);
    else
        ppoStart = N + 1;
        ppoEnd   = N + Nppo;
        ppoSurvived = false(Nppo,1);
        survivedIndex = selected(selected >= ppoStart & selected <= ppoEnd) - N;
        ppoSurvived(survivedIndex) = true;
        ppoSurvivalRate = mean(ppoSurvived);
    end
end
