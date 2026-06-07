function [eliteIndiv,totalFElower] = llSearchCMAES(Problem,ulPopDec,llPopDec,totalFElower)
% Obtain the lower-level response by a basic CMA-ES search.

    lower = Problem.lower(Problem.DU+1:end);
    upper = Problem.upper(Problem.DU+1:end);
    lambda = Problem.N;
    lowerTol = 1e-5;

    if isempty(llPopDec)
        xmean = unifrnd(lower,upper);
    else
        xmean = min(max(llPopDec(1,:),lower),upper);
    end

    lowerCMA = InitializeCMAES(Problem.DL,lower,upper,lambda,xmean);

    initDec = xmean;
    if lambda > 1
        initDec = [initDec;unifrnd(repmat(lower,lambda-1,1),repmat(upper,lambda-1,1))];
    end
    llPopulation = Problem.EvaluationLower([repmat(ulPopDec,size(initDec,1),1),initDec]);

    FElower = length(llPopulation);
    totalFElower = totalFElower + length(llPopulation);
    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
    bestLowerStopFit = min(lowerStopFit);

    while FElower < Problem.maxFElower && bestLowerStopFit > lowerTol
        batchSize = min(lambda,Problem.maxFElower-FElower);
        if batchSize <= 0
            break;
        end

        [llOffDec,llSteps] = SampleCMAES(lowerCMA,batchSize);
        llOffspring = Problem.EvaluationLower([repmat(ulPopDec,batchSize,1),llOffDec]);

        FElower = FElower + length(llOffspring);
        totalFElower = totalFElower + length(llOffspring);

        lowerCMA = UpdateCMAES(lowerCMA,llSteps,llOffDec,CalFitness(Problem.C,llOffspring));

        llPopulation = SelectBestPopulation(Problem,[llPopulation,llOffspring],lambda);
        lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
        bestLowerStopFit = min(lowerStopFit);
    end

    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);
    if all(isinf(lowerStopFit))
        [~,best] = min(CalFitness(Problem.C,llPopulation));
    else
        [~,best] = min(lowerStopFit);
    end
    eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
end

function Population = SelectBestPopulation(Problem,Population,N)
    [~,rank] = sort(CalFitness(Problem.C,Population),'ascend');
    Population = Population(rank(1:min(N,length(rank))));
end

function lowerStopFit = CalLowerStopFitness(Problem,llPopulation)
    PopDec = llPopulation.decs;
    PopObj = llPopulation.objs;
    PopCon = llPopulation.cons;

    N = size(PopDec,1);
    lowerGap = nan(N,1);

    for i = 1 : N
        lowerGap(i) = CalLowerGap(Problem,PopDec(i,:),PopObj(i,:));
    end

    if isempty(PopCon)
        LLCV = zeros(N,1);
    else
        if size(PopCon,2) >= Problem.C + 1
            llCon = PopCon(:,Problem.C+1:end);
        else
            llCon = [];
        end

        if isempty(llCon)
            LLCV = zeros(N,1);
        else
            LLCV = sum(max(0,llCon),2);
        end
    end

    unknown = isnan(lowerGap);
    lowerGap(unknown) = inf;

    feasible = LLCV <= 0;
    lowerStopFit = feasible.*lowerGap + ~feasible.*(LLCV + 1e10);
end

function lowerGap = CalLowerGap(Problem,Dec,Obj)
    problemName = class(Problem);

    if length(Obj) < 2
        lowerGap = nan;
        return;
    end

    FL = Obj(2);

    if ~isprop(Problem,'p') || ~isprop(Problem,'r') || ~isprop(Problem,'q')
        lowerGap = nan;
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
            lowerGap = nan;
            return;
    end

    lowerGap = abs(FL - FLstar);
end
