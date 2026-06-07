function Population = EnvironmentalSelectionUpper(Problem,Population,Offspring)
% Select the best upper-level population from parents and offspring.

    N = length(Population);
    Pool = [Population,Offspring];
    [~,rank] = sort(CalFitness(Problem.C,Pool),'ascend');
    Population = Pool(rank(1:N));
end
