function [LowerFit,lowerGap] = CalOneLowerFitness(Problem,Dec,Obj,Con)
% Calculate lower-level fitness and lower-level gap for one complete solution.

    if length(Obj) >= 2
        FL = Obj(2);
    else
        FL = nan;
    end

    lowerGap = CalOneLowerGap(Problem,Dec,Obj);

    if isempty(Con) || length(Con) < Problem.C + 1
        LLCV = 0;
    else
        llCon = Con(Problem.C+1:end);
        if isempty(llCon)
            LLCV = 0;
        else
            LLCV = sum(max(0,llCon));
        end
    end

    if LLCV <= 0
        LowerFit = FL;
    else
        LowerFit = LLCV + 1e10;
    end
end

function lowerGap = CalOneLowerGap(Problem,Dec,Obj)
% lowerGap = abs(FL - FLstar(xu))

    problemName = class(Problem);

    if length(Obj) < 2
        lowerGap = inf;
        return;
    end

    FL = Obj(2);

    if ~isprop(Problem,'p') || ~isprop(Problem,'r') || ~isprop(Problem,'q')
        lowerGap = inf;
        return;
    end

    xu1 = Dec(1:Problem.p);

    switch problemName
        case {'SMD1','SMD2','SMD3','SMD4','SMD5','SMD6'}
            FLstar = sum(xu1.^2,2);
        case 'SMD7'
            FLstar = sum(xu1.^3,2);
        case 'SMD8'
            FLstar = sum(abs(xu1),2);
        case {'SMD9','SMD10'}
            FLstar = sum(xu1.^2,2);
        case {'SMD11','SMD12'}
            FLstar = sum(xu1.^2,2) + 1;
        otherwise
            lowerGap = inf;
            return;
    end

    lowerGap = abs(FL - FLstar);
end
