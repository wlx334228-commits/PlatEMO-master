function Population = EnvironmentalSelection(Problem,Population,Offspring)


    selected = randperm(length(Population),2);
    Pool     = [Population(selected),Offspring];
    [~,rank] = sort(CalFitness(Problem.C,Pool));
    Population(selected) = Pool(rank(1:length(selected)));
end