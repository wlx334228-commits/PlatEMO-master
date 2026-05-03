function state = BuildState(Problem,Population,lastBest)

    PopDec = Population.decs;
    PopCon = Population.cons;

    ulPop = PopDec(:,1:Problem.DU);
    ulFit = CalFitness(Problem.C,Population);

    N = size(ulPop,1);

    %% 取上层约束和下层约束
    if isempty(PopCon)
        ulCon = zeros(N,1);
        llCon = zeros(N,1);
    else
        if Problem.C > 0
            ulCon = PopCon(:,1:Problem.C);
        else
            ulCon = zeros(N,1);
        end

        if size(PopCon,2) >= Problem.C + 1
            llCon = PopCon(:,Problem.C+1:end);
        else
            llCon = zeros(N,1);
        end

        if isempty(llCon)
            llCon = zeros(N,1);
        end
    end

    bestFit = min(ulFit);
    meanFit = mean(ulFit);

    if isscalar(ulFit)
        stdFit = 0;
    else
        stdFit = std(ulFit);
    end

    if size(ulPop,1) == 1
        divU = 0;
    else
        divU = mean(std(ulPop,0,1));
    end

    feasULRate = mean(all(ulCon <= 0,2));
    feasLLRate = mean(all(llCon <= 0,2));

    if isempty(lastBest)
        improveRate = 0;
    else
        improveRate = lastBest - bestFit;
    end

    [~,rank] = sort(ulFit,'ascend');
    eliteNum = max(2,ceil(0.2*numel(rank)));
    eliteIdx = rank(1:eliteNum);

    eliteFitMean = mean(ulFit(eliteIdx));
    eliteGap = meanFit - eliteFitMean;

    elitePop = ulPop(eliteIdx,:);

    if size(elitePop,1) == 1
        eliteDiv = 0;
    else
        eliteDiv = mean(std(elitePop,0,1));
    end

    meanCV = mean(sum(max(0,ulCon),2) + sum(max(0,llCon),2));

    state = [signedLog(bestFit), ...
             signedLog(meanFit), ...
             log1p(max(0,stdFit)), ...
             log1p(max(0,divU)), ...
             2*feasULRate - 1, ...
             2*feasLLRate - 1, ...
             signedLog(improveRate), ...
             signedLog(eliteGap), ...
             log1p(max(0,eliteDiv)), ...
             signedLog(meanCV)];

    state = max(min(state,5),-5);
end

function y = signedLog(x)
    y = sign(x).*log1p(abs(x));
end