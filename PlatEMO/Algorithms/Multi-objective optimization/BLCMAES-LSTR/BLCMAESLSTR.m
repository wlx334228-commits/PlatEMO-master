classdef BLCMAESLSTR < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    % Bilevel CMA-ES with local successful transition reuse.
    methods
        function main(Algorithm,Problem)
            %% Parameters
            upperTol        = 1e-5;
            archiveMaxSize  = 10 * Problem.N;
            neighborK       = 5;
            reuseThreshold  = 0.25 * sqrt(Problem.DU);
            directionNoise  = 0.20;
            maxStepRatio    = 0.20;

            %% Total lower-level function evaluations
            totalFElower = 0;

            %% Generate random upper population
            ulPopDec = unifrnd( ...
                repmat(Problem.lower(1:Problem.DU),Problem.N,1), ...
                repmat(Problem.upper(1:Problem.DU),Problem.N,1));

            llPopDec = zeros(Problem.N,Problem.DL);
            for i = 1 : size(ulPopDec,1)
                [llPopDec(i,:),totalFElower] = llSearchCMAES(Problem,ulPopDec(i,:),[],totalFElower);
            end

            Population = Problem.Evaluation([ulPopDec,llPopDec]);
            Archive = InitializeTransitionArchive(Problem,archiveMaxSize);

            upperCMA = InitializeCMAES( ...
                Problem.DU, ...
                Problem.lower(1:Problem.DU), ...
                Problem.upper(1:Problem.DU), ...
                Problem.N, ...
                GetUpperCMAInitialMean(Problem,Population));

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

                OldPopulation = Population;

                %% 1. Generate upper-level offspring by CMA-ES and LSTR
                [ulBase,~] = SampleCMAES(upperCMA,Problem.N);

                params.neighborK = neighborK;
                params.reuseThreshold = reuseThreshold;
                params.directionNoise = directionNoise;
                params.maxStepRatio = maxStepRatio;
                ulOffDec = GenerateLSTROffspring(ulBase,Archive,Problem,params);
                upperSteps = CMAESStepsFromDec(upperCMA,ulOffDec);

                %% 2. Warm-start lower-level CMA-ES from nearest current population member
                AllDec = Population.decs;
                AllUL = AllDec(:,1:Problem.DU);
                [~,closest] = min(pdist2(ulOffDec,AllUL),[],2);

                llOffDec = zeros(size(ulOffDec,1),Problem.DL);
                for i = 1 : size(ulOffDec,1)
                    [llOffDec(i,:),totalFElower] = llSearchCMAES( ...
                        Problem, ...
                        ulOffDec(i,:), ...
                        AllDec(closest(i),Problem.DU+1:end), ...
                        totalFElower);
                end

                %% 3. Evaluate offspring and update LSTR archive
                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);
                Archive = UpdateTransitionArchive(Problem,Archive,OldPopulation,Offspring);

                %% 4. Update upper-level CMA-ES from evaluated offspring
                upperCMA = UpdateCMAES(upperCMA,upperSteps,ulOffDec,CalFitness(Problem.C,Offspring));

                %% 5. Upper-level environmental selection
                Population = EnvironmentalSelectionUpper(Problem,Population,Offspring);

                %% 6. Record best upper/lower information
                Fitness = CalFitness(Problem.C,Population);
                [UpperFit,best] = min(Fitness);

                bestDec = Population(best).dec;
                bestObj = Population(best).obj;
                bestCon = Population(best).con;
                [LowerFit,lowerGap] = CalOneLowerFitness(Problem,bestDec,bestObj,bestCon);

                fprintf(['BLCMAESLSTR Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | ', ...
                    'UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e | Archive=%d | Sigma=%.3e\n'], ...
                    gen,Problem.FE,totalFElower,UpperFit,LowerFit,lowerGap, ...
                    size(Archive.X0,1),mean(upperCMA.sigma));

                if abs(UpperFit) <= upperTol
                    upperReached = true;
                    fprintf('BLCMAESLSTR upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        UpperFit,Problem.FE,totalFElower);
                end

                gen = gen + 1;
            end

            writeRunRecord();

            function writeRunRecord()
                if ~runRecordWritten
                    AppendBLCMAESLSTRRunFERecord(Problem,Problem.FE,totalFElower,max(0,gen-1),upperReached);
                    runRecordWritten = true;
                end
            end
        end
    end
end

function xmean = GetUpperCMAInitialMean(Problem,Population)
    PopDec = Population.decs;
    ulPop = PopDec(:,1:Problem.DU);
    fitness = CalFitness(Problem.C,Population);
    [~,rank] = sort(fitness,'ascend');
    eliteNum = max(2,ceil(0.5*size(ulPop,1)));
    xmean = mean(ulPop(rank(1:eliteNum),:),1);
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
