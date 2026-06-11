function Archive = UpdateTransitionArchive(Problem,Archive,OldPopulation,Offspring)
% Store successful upper-level transitions from nearest old solutions.

    oldDec = OldPopulation.decs;
    oldUL = oldDec(:,1:Problem.DU);
    oldObj = UpperObjective(OldPopulation);
    oldCV  = ConstraintViolation(OldPopulation);
    oldFeasible = oldCV <= 0;

    offDec = Offspring.decs;
    offUL = offDec(:,1:Problem.DU);
    offObj = UpperObjective(Offspring);
    offCV  = ConstraintViolation(Offspring);
    offFeasible = offCV <= 0;

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
        d = offUL(i,:) - oldUL(ref,:);

        successful = false;
        relImprove = 0;

        if ~oldFeasible(ref) && offFeasible(i)
            successful = true;
            relImprove = (oldCV(ref) - offCV(i)) / (abs(oldCV(ref)) + 1e-8);
        elseif oldFeasible(ref) && offFeasible(i)
            improve = oldObj(ref) - offObj(i);
            relImprove = improve / (abs(oldObj(ref)) + 1e-8);
            successful = improve > 1e-12 && relImprove > 1e-10;
        end

        if successful && relImprove > 1e-10 && any(abs(d) > 1e-12)
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

function Obj = UpperObjective(Population)
% Extract upper-level objective values.

    PopObj = Population.objs;
    Obj = PopObj(:,1);
end

function CV = ConstraintViolation(Population)
% Sum all constraint violations in the evaluated candidate.

    PopObj = Population.objs;
    CV = zeros(size(PopObj,1),1);
    PopCon = Population.cons;
    if isempty(PopCon)
        return;
    end

    PopCon(isnan(PopCon)) = 0;
    CV = sum(max(0,PopCon),2);
end
