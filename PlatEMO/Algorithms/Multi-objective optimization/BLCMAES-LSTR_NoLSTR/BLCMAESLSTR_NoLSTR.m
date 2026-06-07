classdef BLCMAESLSTR_NoLSTR < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    % Ablation of BLCMAES-LSTR without local successful transition reuse.
    methods
        function main(Algorithm,Problem)
            %% Parameters
            upperTol = 1e-6;

            %% Total lower-level function evaluations
            totalFElower = 0;

            %% Initialize global CMA-ES over both upper and lower variables
            globalCMA = InitializeCMAES(Problem.D,Problem.lower,Problem.upper,[],[]);

            %% Generate initial upper population from global CMA-ES
            initN = min(globalCMA.lambda,Problem.maxFE);
            [initDec,~] = SampleCMAES(globalCMA,initN);
            ulPopDec = initDec(:,1:Problem.DU);
            llPopDec = zeros(size(ulPopDec,1),Problem.DL);
            for i = 1 : size(ulPopDec,1)
                [llPopDec(i,:),totalFElower] = llSearchCMAES(Problem,ulPopDec(i,:),globalCMA,totalFElower);
            end

            Population = Problem.Evaluation([ulPopDec,llPopDec]);

            upperReached = false;
            gen = 1;
            runRecordWritten = false;

            %% Upper-level optimization
            while true
                try
                    nofinish = Algorithm.NotTerminated(Population);
                catch err
                    if strcmp(err.identifier,'PlatEMO:Termination')
                        writeRunRecord();
                    end
                    rethrow(err);
                end

                if ~nofinish || upperReached
                    break;
                end

                %% 1. Generate upper-level offspring by global CMA-ES
                remainingUpperFE = Problem.maxFE - Problem.FE;
                if remainingUpperFE <= 0
                    break;
                end
                offspringN = min(globalCMA.lambda,remainingUpperFE);
                [baseDec,~] = SampleCMAES(globalCMA,offspringN);
                ulOffDec = baseDec(:,1:Problem.DU);

                %% 2. Search lower-level responses by lower marginal CMA-ES
                llOffDec = zeros(size(ulOffDec,1),Problem.DL);
                for i = 1 : size(ulOffDec,1)
                    [llOffDec(i,:),totalFElower] = llSearchCMAES( ...
                        Problem, ...
                        ulOffDec(i,:), ...
                        globalCMA, ...
                        totalFElower);
                end

                %% 3. Evaluate offspring
                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);

                %% 4. Upper-level environmental selection
                Population = EnvironmentalSelectionUpper(Problem,Population,Offspring);

                %% 5. Update global CMA-ES from selected parent-offspring elites
                globalCMA = UpdateCMAES(globalCMA,Population.decs,CalFitness(Problem.C,Population));

                %% 6. Record best upper/lower information
                Fitness = CalFitness(Problem.C,Population);
                [UpperFit,best] = min(Fitness);

                bestDec = Population(best).dec;
                bestObj = Population(best).obj;
                bestCon = Population(best).con;
                [LowerFit,lowerGap] = CalOneLowerFitness(Problem,bestDec,bestObj,bestCon);

                fprintf(['BLCMAESLSTR_NoLSTR Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | ', ...
                    'UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e | Sigma=%.3e\n'], ...
                    gen,Problem.FE,totalFElower,UpperFit,LowerFit,lowerGap,globalCMA.sigma);

                if abs(UpperFit) <= upperTol
                    upperReached = true;
                    fprintf('BLCMAESLSTR_NoLSTR upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        UpperFit,Problem.FE,totalFElower);
                end

                gen = gen + 1;
            end

            writeRunRecord();

            function writeRunRecord()
                if ~runRecordWritten
                    AppendBLCMAESLSTRNoLSTRRunFERecord(Problem,Problem.FE,totalFElower,max(0,gen-1),upperReached);
                    runRecordWritten = true;
                end
            end
        end
    end
end

function [LowerFit,lowerGap] = CalOneLowerFitness(Problem,Dec,Obj,Con)
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
    if length(Obj) < 2
        lowerGap = inf;
        return;
    end

    FL = Obj(2);
    if ~isprop(Problem,'p')
        lowerGap = inf;
        return;
    end
    xu1 = Dec(1:Problem.p);
    problemName = class(Problem);

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
            FLstar = nan;
    end

    if isnan(FLstar)
        lowerGap = inf;
    else
        lowerGap = abs(FL - FLstar);
    end
end
