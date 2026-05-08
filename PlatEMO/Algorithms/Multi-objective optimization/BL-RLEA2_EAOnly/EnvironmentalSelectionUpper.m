function Population = EnvironmentalSelectionUpper(Problem,Population,Offspring)
% Generational environmental selection for upper-level population

    N = length(Population);

    Pool = [Population,Offspring];

    [~,rank] = sort(CalFitness(Problem.C,Pool));

    Population = Pool(rank(1:N));
end