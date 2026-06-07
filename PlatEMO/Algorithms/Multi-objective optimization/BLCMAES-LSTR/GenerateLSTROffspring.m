function ulOffDec = GenerateLSTROffspring(ulBaseDec,Archive,Problem,params)
% Correct CMA-ES upper offspring by locally reusing successful transitions.

    if isempty(ulBaseDec)
        ulOffDec = zeros(0,Problem.DU);
        return;
    end

    DU = Problem.DU;
    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);
    range = upper - lower;
    range(range < 1e-12) = 1;

    ulOffDec = ulBaseDec;
    for i = 1 : size(ulBaseDec,1)
        [direction,usable] = QueryLocalTransitions(ulBaseDec(i,:),Archive,Problem,params);
        if usable
            maxStep = params.maxStepRatio .* range;
            direction = min(max(direction,-maxStep),maxStep);
            ulOffDec(i,:) = ulBaseDec(i,:) + direction;
        end
    end

    ulOffDec = RepairBounds(ulOffDec,lower,upper);
end
