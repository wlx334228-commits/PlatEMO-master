function eliteIndiv = llSearch(Problem,ulPopDec,llPopDec)

    %% Lower level population initialization
    llPopDec     = [llPopDec;unifrnd(repmat(Problem.lower(Problem.DU+1:end),Problem.N-size(llPopDec,1),1),repmat(Problem.upper(Problem.DU+1:end),Problem.N-size(llPopDec,1),1))];
    llPopulation = Problem.EvaluationLower([repmat(ulPopDec,Problem.N,1),llPopDec]);
    FElower      = 0;
    %% Optimization
    while FElower < Problem.maxFElower
        % Select parents and generate offspring
        MatingPool  = TournamentSelection(2,3,CalFitness(Problem.C,llPopulation));
        ParentDec   = llPopulation(MatingPool).decs;
        llOffDec    = OperatorPCX(ParentDec(:,Problem.DU+1:end),Problem.lower(Problem.DU+1:end),Problem.upper(Problem.DU+1:end));
        llOffspring = Problem.EvaluationLower([repmat(ulPopDec,size(llOffDec,1),1),llOffDec]);
        FElower     = FElower + length(llOffspring);
        % Select r members with better adaptability
        llPopulation = EnvironmentalSelection(Problem,llPopulation,llOffspring);
    end
    [~,best]   = min(CalFitness(Problem.C,llPopulation));
    eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
end