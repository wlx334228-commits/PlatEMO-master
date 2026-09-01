function [Archive,newKnowledgeN] = UpdateTransitionArchive(Problem,Archive,HistoryWindow,CurrentRecords,~,params,~,gen)
% Store local transitions from evaluated worse upper points to better ones.

    newKnowledgeN = 0;
    if nargin < 8
        gen = 0;
    end
    if isempty(HistoryWindow) || isempty(HistoryWindow.X) || isempty(CurrentRecords.X)
        return;
    end

    radius = GetParam(params,'archiveThreshold',GetParam(params,'archiveRadius',inf));
    if radius <= 0
        return;
    end

    DU = Problem.DU;
    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;

    for i = 1 : size(CurrentRecords.X,1)
        cur = GetRecord(CurrentRecords,i);
        for j = 1 : size(HistoryWindow.X,1)
            hist = GetRecord(HistoryWindow,j);
            dPair = norm((cur.X - hist.X) ./ range) / sqrt(DU);
            if dPair > radius
                continue;
            end

            [Archive,added] = TryAppendFeasibilityTransition(Archive,hist,cur,dPair,params,gen);
            newKnowledgeN = newKnowledgeN + added;

            [Archive,added] = TryAppendObjectiveTransition(Archive,hist,cur,dPair,params,gen);
            newKnowledgeN = newKnowledgeN + added;

            [Archive,added] = TryAppendBalancedTransition(Archive,hist,cur,dPair,params,gen);
            newKnowledgeN = newKnowledgeN + added;
        end
    end
end

function record = GetRecord(Records,idx)
    record.X = Records.X(idx,:);
    record.F = Records.F(idx);
    record.CV = Records.CV(idx);
    record.Fit = Records.Fit(idx);
    record.Feasible = Records.Feasible(idx);
    record.RF = Records.RF(idx);
    record.Gen = Records.Gen(idx);
end

function [Archive,added] = TryAppendFeasibilityTransition(Archive,A,B,dPair,params,gen)
    added = 0;
    [source,target,deltaCV] = OrientByDecrease(A,B,'CV',params.cvTol);
    if isempty(source)
        return;
    end

    deltaF = source.F - target.F;
    I = deltaCV / (abs(source.CV) + params.cvTol);
    if I <= 0
        return;
    end

    Archive = AppendTransition(Archive,'Feasibility',source,target,dPair,deltaF,deltaCV,I,gen);
    added = 1;
end

function [Archive,added] = TryAppendObjectiveTransition(Archive,A,B,dPair,params,gen)
    added = 0;
    if ~(A.Feasible && B.Feasible)
        return;
    end

    [source,target,deltaF] = OrientByDecrease(A,B,'F',params.objTol);
    if isempty(source)
        return;
    end

    deltaCV = source.CV - target.CV;
    I = deltaF / (abs(source.F) + 1);
    if I <= 0
        return;
    end

    Archive = AppendTransition(Archive,'Objective',source,target,dPair,deltaF,deltaCV,I,gen);
    added = 1;
end

function [Archive,added] = TryAppendBalancedTransition(Archive,A,B,dPair,params,gen)
    added = 0;
    if IsBetterByConstraintPriority(B,A,params)
        source = A;
        target = B;
    elseif IsBetterByConstraintPriority(A,B,params)
        source = B;
        target = A;
    else
        return;
    end

    deltaF = source.F - target.F;
    deltaCV = source.CV - target.CV;
    if source.CV > params.cvTol && deltaCV > params.cvTol
        I = deltaCV / (abs(source.CV) + params.cvTol);
    elseif source.Feasible && target.Feasible && deltaF > params.objTol
        I = deltaF / (abs(source.F) + 1);
    else
        deltaFit = source.Fit - target.Fit;
        I = deltaFit / (abs(source.Fit) + 1);
    end
    if I <= 0
        return;
    end

    Archive = AppendTransition(Archive,'Balanced',source,target,dPair,deltaF,deltaCV,I,gen);
    added = 1;
end

function [source,target,delta] = OrientByDecrease(A,B,fieldName,tol)
    source = [];
    target = [];
    delta = 0;
    a = A.(fieldName);
    b = B.(fieldName);
    if b < a - tol
        source = A;
        target = B;
        delta = a - b;
    elseif a < b - tol
        source = B;
        target = A;
        delta = b - a;
    end
end

function better = IsBetterByConstraintPriority(A,B,params)
    if A.Feasible && B.Feasible
        better = A.F < B.F - params.objTol;
    elseif A.Feasible && ~B.Feasible
        better = true;
    elseif ~A.Feasible && B.Feasible
        better = false;
    else
        better = A.CV < B.CV - params.cvTol;
    end
end

function Archive = AppendTransition(Archive,fieldName,source,target,dPair,deltaF,deltaCV,I,gen)
    SubArchive = Archive.(fieldName);
    Archive.Count = Archive.Count + 1;
    d = target.X - source.X;

    SubArchive.XSource(end+1,:) = source.X;
    SubArchive.XTarget(end+1,:) = target.X;
    SubArchive.X0(end+1,:) = source.X;
    SubArchive.D(end+1,:) = d;
    SubArchive.DPair(end+1,1) = dPair;
    SubArchive.DeltaF(end+1,1) = deltaF;
    SubArchive.DeltaCV(end+1,1) = deltaCV;
    SubArchive.I(end+1,1) = I;
    SubArchive.W(end+1,1) = max(I,eps);
    SubArchive.Gen(end+1,1) = gen;
    SubArchive = TrimTransitionSubArchive(SubArchive);
    Archive.(fieldName) = SubArchive;
end

function SubArchive = TrimTransitionSubArchive(SubArchive)
    if size(SubArchive.XSource,1) > SubArchive.MaxSize
        keep = size(SubArchive.XSource,1) - SubArchive.MaxSize + 1 : size(SubArchive.XSource,1);
        SubArchive.XSource = SubArchive.XSource(keep,:);
        SubArchive.XTarget = SubArchive.XTarget(keep,:);
        SubArchive.X0 = SubArchive.X0(keep,:);
        SubArchive.D = SubArchive.D(keep,:);
        SubArchive.DPair = SubArchive.DPair(keep,:);
        SubArchive.DeltaF = SubArchive.DeltaF(keep,:);
        SubArchive.DeltaCV = SubArchive.DeltaCV(keep,:);
        SubArchive.I = SubArchive.I(keep,:);
        SubArchive.W = SubArchive.W(keep,:);
        SubArchive.Gen = SubArchive.Gen(keep,:);
    end
end

function value = GetParam(params,name,defaultValue)
    if isfield(params,name)
        value = params.(name);
    else
        value = defaultValue;
    end
end
