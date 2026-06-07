function Archive = UpdateTransitionArchive(Problem,Archive,OldPopulation,Offspring)
% Store successful upper-level transitions from nearest old solutions.

    oldDec = OldPopulation.decs;
    oldUL = oldDec(:,1:Problem.DU);
    oldFit = CalFitness(Problem.C,OldPopulation);
    oldFit = oldFit(:);

    offDec = Offspring.decs;
    offUL = offDec(:,1:Problem.DU);
    offFit = CalFitness(Problem.C,Offspring);
    offFit = offFit(:);

    if isempty(offUL)
        return;
    end

    [~,closest] = min(pdist2(offUL,oldUL),[],2);

    newX0 = zeros(0,Problem.DU);
    newD = zeros(0,Problem.DU);
    newW = zeros(0,1);
    newGen = zeros(0,1);

    for i = 1 : size(offUL,1)
        ref = closest(i);
        improve = oldFit(ref) - offFit(i);
        relImprove = improve / (abs(oldFit(ref)) + 1e-8);
        d = offUL(i,:) - oldUL(ref,:);

        if relImprove > 1e-10 && any(abs(d) > 1e-12)
            newX0(end+1,:) = oldUL(ref,:);
            newD(end+1,:) = d;
            newW(end+1,1) = relImprove;
            newGen(end+1,1) = Archive.Count + 1;
            Archive.Count = Archive.Count + 1;
        end
    end

    if isempty(newX0)
        return;
    end

    Archive.X0 = [Archive.X0;newX0];
    Archive.D = [Archive.D;newD];
    Archive.W = [Archive.W;newW];
    Archive.Gen = [Archive.Gen;newGen];

    if size(Archive.X0,1) > Archive.MaxSize
        keep = size(Archive.X0,1) - Archive.MaxSize + 1 : size(Archive.X0,1);
        Archive.X0 = Archive.X0(keep,:);
        Archive.D = Archive.D(keep,:);
        Archive.W = Archive.W(keep,:);
        Archive.Gen = Archive.Gen(keep,:);
    end
end
