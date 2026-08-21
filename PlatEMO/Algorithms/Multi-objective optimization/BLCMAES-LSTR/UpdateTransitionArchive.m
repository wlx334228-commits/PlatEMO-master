function Archive = UpdateTransitionArchive(Problem,Archive,OldPopulation,Offspring,BI,params)
% Store constraint-aware successful upper-level transitions.

    if nargin < 6 || isempty(params)
        params.cvTol = 1e-8;
        params.objTol = 1e-12;
    end

    oldDec = OldPopulation.decs;
    oldUL = oldDec(:,1:Problem.DU);
    oldObj = UpperObjective(OldPopulation);
    oldCV  = ConstraintViolation(OldPopulation,BI);
    oldFeasible = oldCV <= params.cvTol;

    offDec = Offspring.decs;
    offUL = offDec(:,1:Problem.DU);
    offObj = UpperObjective(Offspring);
    offCV  = ConstraintViolation(Offspring,BI);
    offFeasible = offCV <= params.cvTol;

    if isempty(offUL) || isempty(oldUL)
        return;
    end

    [~,closest] = min(pdist2(offUL,oldUL),[],2);

    for i = 1 : size(offUL,1)
        ref = closest(i);
        d = offUL(i,:) - oldUL(ref,:);
        if ~any(abs(d) > 1e-12)
            continue;
        end

        cvImprove = oldCV(ref) > params.cvTol && offCV(i) < oldCV(ref) - params.cvTol;
        objImprove = oldFeasible(ref) && offFeasible(i) && offObj(i) < oldObj(ref) - params.objTol;

        if cvImprove
            w = (oldCV(ref) - offCV(i)) / (abs(oldCV(ref)) + params.cvTol);
            Archive = AppendTransition(Archive,'Feasibility',oldUL(ref,:),d,max(w,params.objTol));
        end

        if objImprove
            w = (oldObj(ref) - offObj(i)) / (abs(oldObj(ref)) + 1);
            Archive = AppendTransition(Archive,'Objective',oldUL(ref,:),d,max(w,params.objTol));
        end
    end
end

function Archive = AppendTransition(Archive,fieldName,x0,d,w)
    SubArchive = Archive.(fieldName);
    Archive.Count = Archive.Count + 1;
    SubArchive.X0(end+1,:) = x0;
    SubArchive.D(end+1,:) = d;
    SubArchive.W(end+1,1) = w;
    SubArchive.Gen(end+1,1) = Archive.Count;
    SubArchive = TrimTransitionSubArchive(SubArchive);
    Archive.(fieldName) = SubArchive;
end

function SubArchive = TrimTransitionSubArchive(SubArchive)
    if size(SubArchive.X0,1) > SubArchive.MaxSize
        keep = size(SubArchive.X0,1) - SubArchive.MaxSize + 1 : size(SubArchive.X0,1);
        SubArchive.X0 = SubArchive.X0(keep,:);
        SubArchive.D = SubArchive.D(keep,:);
        SubArchive.W = SubArchive.W(keep,:);
        SubArchive.Gen = SubArchive.Gen(keep,:);
    end
end

function Obj = UpperObjective(Population)
% Extract upper-level objective values.

    PopObj = Population.objs;
    Obj = PopObj(:,1);
end

function CV = ConstraintViolation(Population,BI)
% Constraint violation under the same constraint scope used by upper selection.

    PopObj = Population.objs;
    CV = zeros(size(PopObj,1),1);
    PopCon = Population.cons;
    if isempty(PopCon)
        return;
    end

    if nargin >= 2 && ~isempty(BI) && BI.isLowerLevelConstraintsIncludedInUpperLevel
        upperConN = min(BI.upperConN,size(PopCon,2));
        if upperConN < 1
            return;
        end
        PopCon = PopCon(:,1:upperConN);
    end

    PopCon(isnan(PopCon)) = 0;
    CV = sum(max(0,PopCon),2);
end
