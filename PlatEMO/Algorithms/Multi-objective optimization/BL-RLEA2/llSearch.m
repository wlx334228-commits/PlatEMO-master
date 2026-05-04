% function eliteIndiv = llSearch(Problem,ulPopDec,llPopDec)
% 
%     %% Lower level population initialization
%     llPopDec     = [llPopDec;unifrnd(repmat(Problem.lower(Problem.DU+1:end),Problem.N-size(llPopDec,1),1),repmat(Problem.upper(Problem.DU+1:end),Problem.N-size(llPopDec,1),1))];
%     llPopulation = Problem.EvaluationLower([repmat(ulPopDec,Problem.N,1),llPopDec]);
%     FElower      = 0;
%     %% Optimization
%     while FElower < Problem.maxFElower
%         % Select parents and generate offspring
%         MatingPool  = TournamentSelection(2,3,CalFitness(Problem.C,llPopulation));
%         ParentDec   = llPopulation(MatingPool).decs;
%         llOffDec    = OperatorPCX(ParentDec(:,Problem.DU+1:end),Problem.lower(Problem.DU+1:end),Problem.upper(Problem.DU+1:end));
%         llOffspring = Problem.EvaluationLower([repmat(ulPopDec,size(llOffDec,1),1),llOffDec]);
%         FElower     = FElower + length(llOffspring);
%         % Select r members with better adaptability
%         llPopulation = EnvironmentalSelection(Problem,llPopulation,llOffspring);
%     end
%     [~,best]   = min(CalFitness(Problem.C,llPopulation));
%     eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
% end
function [eliteIndiv,totalFElower] = llSearch(Problem,ulPopDec,llPopDec,totalFElower)
% Obtain the lower-level member corresponding to the upper-level member.
%
% 下层搜索逻辑：
% 1. 固定上层变量 ulPopDec
% 2. 用 Problem.EvaluationLower() 评价下层种群
% 3. 用 CalFitness() 进行下层选择
% 4. 用 lowerStopFit 判断是否达到下层精度 1e-6
%
% lowerStopFit 不是原始 FL，而是：
%   如果下层可行：
%       lowerStopFit = lowerGap = abs(FL - FLstar(xu))
%   如果下层不可行：
%       lowerStopFit = LLCV + 1e10



    %% Lower level population initialization
    llPopDec = [llPopDec;unifrnd( ...
        repmat(Problem.lower(Problem.DU+1:end),Problem.N-size(llPopDec,1),1), ...
        repmat(Problem.upper(Problem.DU+1:end),Problem.N-size(llPopDec,1),1))];

    %% Lower-level evaluation by PlatEMO
    llPopulation = Problem.EvaluationLower([repmat(ulPopDec,Problem.N,1),llPopDec]);

    FElower = length(llPopulation);
    totalFElower = totalFElower + length(llPopulation);
    lowerTol = 1e-6;

    %% Check whether initial lower population already satisfies accuracy
    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);

    if min(lowerStopFit) <= lowerTol
        [~,best] = min(lowerStopFit);
        eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
        return;
    end

    %% Lower-level optimization
    while FElower < Problem.maxFElower

        %% Select parents according to original lower-level fitness
        % CalFitness() 会自动识别这是下层种群：
        % 因为 EvaluationLower() 中 PopObj(:,1)=nan
        MatingPool = TournamentSelection(2,3,CalFitness(Problem.C,llPopulation));
        ParentDec  = llPopulation(MatingPool).decs;

        %% Generate lower-level offspring
        llOffDec = OperatorPCX( ...
            ParentDec(:,Problem.DU+1:end), ...
            Problem.lower(Problem.DU+1:end), ...
            Problem.upper(Problem.DU+1:end));

        %% Evaluate lower-level offspring by PlatEMO
        llOffspring = Problem.EvaluationLower( ...
            [repmat(ulPopDec,size(llOffDec,1),1),llOffDec]);

        FElower = FElower + length(llOffspring);
        totalFElower = totalFElower + length(llOffspring);

        %% Environmental selection still uses original CalFitness()
        llPopulation = EnvironmentalSelection(Problem,llPopulation,llOffspring);

        %% Check lower-level accuracy
        lowerStopFit = CalLowerStopFitness(Problem,llPopulation);

        if min(lowerStopFit) <= lowerTol
            break;
        end
    end

    %% Return best lower-level solution
    lowerStopFit = CalLowerStopFitness(Problem,llPopulation);

    if all(isinf(lowerStopFit))
        % 如果当前问题没有写 lowerGap 公式，则退回原始下层适应度
        [~,best] = min(CalFitness(Problem.C,llPopulation));
    else
        [~,best] = min(lowerStopFit);
    end

    eliteIndiv = llPopulation(best).dec(Problem.DU+1:end);
end


function lowerStopFit = CalLowerStopFitness(Problem,llPopulation)
% 计算下层截止用的“下层适应度”
%
% 这个函数只用于判断 llSearch 是否可以提前停止。
% 它不替代 CalFitness() 的选择功能。
%
% lowerStopFit =
%   lowerGap,        if lower-level constraints are feasible
%   LLCV + 1e10,     if lower-level constraints are infeasible
%
% lowerGap = abs(FL - FLstar(xu))

    PopDec = llPopulation.decs;
    PopObj = llPopulation.objs;
    PopCon = llPopulation.cons;

    N = size(PopDec,1);

    lowerGap = nan(N,1);

    for i = 1:N
        lowerGap(i) = CalLowerGap(Problem,PopDec(i,:),PopObj(i,:));
    end

    %% Calculate lower-level constraint violation
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

    %% Unknown lowerGap formula: do not allow early stopping
    unknown = isnan(lowerGap);
    lowerGap(unknown) = inf;

    Feasible = LLCV <= 0;

    lowerStopFit = Feasible.*lowerGap + ~Feasible.*(LLCV + 1e10);
end


function lowerGap = CalLowerGap(Problem,Dec,Obj)
% Calculate lower-level objective accuracy:
%
% lowerGap = abs(FL - FLstar(xu))
%
% FL is obtained from Problem.EvaluationLower().
% FLstar is the theoretical lower-level optimum value under fixed xu.

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

    %% Split variables
    xu1 = Dec(1:Problem.p);
    xu2 = Dec(Problem.p+1:Problem.p+Problem.r);

    xl1 = Dec(Problem.p+Problem.r+1 : Problem.p+Problem.r+Problem.q);
    xl2 = Dec(Problem.p+Problem.r+Problem.q+1 : end);

    switch problemName
        case {'SMD1','SMD2','SMD3','SMD4','SMD5','SMD6'}
            FLstar = sum(xu1.^2,2);

        case 'SMD7'
            FLstar = sum(xu1.^3,2);

        case 'SMD8'
            FLstar = sum(abs(xu1),2);

        case {'SMD9','SMD10'}
            FLstar = sum(xu1.^2,2);

        case 'SMD11'
            FLstar = sum(xu1.^2,2) + 1;

        case 'SMD12'
            FLstar = sum(xu1.^2,2) + 1;

        otherwise
            lowerGap = nan;
            return;
    end

    lowerGap = abs(FL - FLstar);
end