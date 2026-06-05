classdef BLLSTR < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    % Local Successful Transition Reuse for evolutionary bilevel optimization.
    methods
        function main(Algorithm,Problem)
            %% Parameters
            guideRatio      = 0.5;
            sigma0          = 1;
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
                [llPopDec(i,:),totalFElower] = llSearch(Problem,ulPopDec(i,:),[],totalFElower);
            end

            Population = Problem.Evaluation([ulPopDec,llPopDec]);
            Archive = InitializeTransitionArchive(Problem,archiveMaxSize);

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
                Fitness = CalFitness(Problem.C,Population);

                %% 1. Generate upper-level offspring
                Noff = Problem.N;
                Nguided = floor(guideRatio * Noff);
                Nea = Noff - Nguided;

                [ThetaBase,~] = BuildEliteTheta(Problem,Population,sigma0);
                baseGuide = GenerateUpperOffspring(ThetaBase,Problem,Nguided);

                params.neighborK = neighborK;
                params.reuseThreshold = reuseThreshold;
                params.directionNoise = directionNoise;
                params.maxStepRatio = maxStepRatio;
                ulGuide = GenerateLSTROffspring(baseGuide,Archive,Problem,params);

                if Nea > 0
                    MatingPool = TournamentSelection(2,Nea,Fitness);
                    ParentDec = Population(MatingPool).decs;
                    ulEA = OperatorSBXPM( ...
                        ParentDec(:,1:Problem.DU), ...
                        Problem.lower(1:Problem.DU), ...
                        Problem.upper(1:Problem.DU));
                else
                    ulEA = zeros(0,Problem.DU);
                end

                ulOffDec = [ulGuide;ulEA];

                %% 2. Warm-start lower-level search from nearest current population member
                AllDec = Population.decs;
                AllUL = AllDec(:,1:Problem.DU);
                [~,closest] = min(pdist2(ulOffDec,AllUL),[],2);

                llOffDec = zeros(size(ulOffDec,1),Problem.DL);
                for i = 1 : size(ulOffDec,1)
                    [llOffDec(i,:),totalFElower] = llSearch( ...
                        Problem, ...
                        ulOffDec(i,:), ...
                        AllDec(closest(i),Problem.DU+1:end), ...
                        totalFElower);
                end

                %% 3. Evaluate offspring and update successful transition archive
                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);
                Archive = UpdateTransitionArchive(Problem,Archive,OldPopulation,Offspring);

                %% 4. Upper-level environmental selection
                Population = EnvironmentalSelectionUpper(Problem,Population,Offspring);

                %% 5. Record best upper/lower information
                Fitness = CalFitness(Problem.C,Population);
                [UpperFit,best] = min(Fitness);

                bestDec = Population(best).dec;
                bestObj = Population(best).obj;
                bestCon = Population(best).con;
                [LowerFit,lowerGap] = CalOneLowerFitness(Problem,bestDec,bestObj,bestCon);

                fprintf(['BLLSTR Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | ', ...
                    'UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e | Archive=%d\n'], ...
                    gen,Problem.FE,totalFElower,UpperFit,LowerFit,lowerGap,size(Archive.X0,1));

                if abs(UpperFit) <= upperTol
                    upperReached = true;
                    fprintf('BLLSTR upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        UpperFit,Problem.FE,totalFElower);
                end

                gen = gen + 1;
            end

            writeRunRecord();

            function writeRunRecord()
                if ~runRecordWritten
                    AppendBLLSTRRunFERecord(Problem,Problem.FE,totalFElower,max(0,gen-1),upperReached);
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
