function ulGuide = GenerateLSTROffspring(baseGuide,Archive,Problem,params)
% Generate upper offspring by locally reusing successful transitions.

    if isempty(baseGuide)
        ulGuide = zeros(0,Problem.DU);
        return;
    end

    DU = Problem.DU;
    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);
    range = upper - lower;
    range(range < 1e-12) = 1;

    ulGuide = baseGuide;

    for i = 1 : size(baseGuide,1)
        [direction,usable] = QueryLocalTransitions(baseGuide(i,:),Archive,Problem,params);
        if usable
            maxStep = params.maxStepRatio .* range;
            direction = min(max(direction,-maxStep),maxStep);
            ulGuide(i,:) = baseGuide(i,:) + direction;
        end
    end

    ulGuide = min(max(ulGuide,repmat(lower,size(ulGuide,1),1)), ...
        repmat(upper,size(ulGuide,1),1));
end
