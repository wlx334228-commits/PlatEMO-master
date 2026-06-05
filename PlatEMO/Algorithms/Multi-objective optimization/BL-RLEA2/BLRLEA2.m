classdef BLRLEA2 < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    methods
        function main(Algorithm,Problem)
            %% PPO parameters
            sigma0 = 1;
            ppoRatio = 0.25;
            stateDim = 2 * Problem.DU;
            actionDim = Problem.DU;
            hiddenDim = 32;
            learnRate = 1e-3;
            perturbEta = 0.2;

            rolloutUpdateGap = 3;
            rolloutGenCount  = 0;
            rolloutBuffer    = InitializePPOBuffer();

            Actor  = InitializeActorNetwork(stateDim,actionDim,hiddenDim,learnRate);
            Critic = InitializeCriticNetwork(stateDim,hiddenDim,learnRate);

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

            upperTol = 1e-5;
            upperReached = false;

            %% Debug log
            gen = 1;
            fid = fopen('BLRLEA2_debug_log.txt','w');
            logCleaner = onCleanup(@()CloseLogFile(fid));
            fprintf(fid,'Gen\tUpperFE\tTotalLowerFE\tUpperFit\tLowerFit\tlowerGap\n');

            %% UL Optimization
            while Algorithm.NotTerminated(Population)
                if upperReached
                    break;
                end

                %% 1. Generate upper-level offspring by elite anchors, PPO perturbation, and SBX+PM
                Fitness = CalFitness(Problem.C,Population);
                Noff = Problem.N;
                Nbase = floor(ppoRatio * Noff);
                Nppo  = Nbase;
                Nea   = Noff - Nbase - Nppo;

                [ThetaBase,eliteInfo] = BuildEliteTheta(Problem,Population,sigma0);
                basePPO = GenerateUpperOffspring(ThetaBase,Problem,Nbase);

                statesPPO      = zeros(Nppo,stateDim);
                actionsPPO     = zeros(Nppo,actionDim);
                logProbsPPO    = zeros(Nppo,1);
                valuesPPO      = zeros(Nppo,1);
                actionMeansPPO = zeros(Nppo,actionDim);
                actionStdsPPO  = zeros(Nppo,actionDim);
                deltaPPO       = zeros(Nppo,actionDim);
                ulPPO          = zeros(Nppo,Problem.DU);

                for i = 1 : Nppo
                    statesPPO(i,:) = BuildPerturbState( ...
                        Problem,basePPO(i,:),eliteInfo);
                    valuesPPO(i) = CriticForward(Critic,statesPPO(i,:));
                    [actionsPPO(i,:),logProbsPPO(i),actionMeansPPO(i,:),actionStdsPPO(i,:)] = ...
                        ActorForward(Actor,statesPPO(i,:));

                    deltaPPO(i,:) = perturbEta .* tanh(actionsPPO(i,:)) .* eliteInfo.perturbScale;
                    ulPPO(i,:) = basePPO(i,:) + deltaPPO(i,:);
                end

                lowerUL = repmat(Problem.lower(1:Problem.DU),Nppo,1);
                upperUL = repmat(Problem.upper(1:Problem.DU),Nppo,1);
                ulPPO = min(max(ulPPO,lowerUL),upperUL);

                if Nea > 0
                    MatingPool = TournamentSelection(2,Nea,Fitness);
                    ParentDec  = Population(MatingPool).decs;
                    ulEA = OperatorSBXPM( ...
                        ParentDec(:,1:Problem.DU), ...
                        Problem.lower(1:Problem.DU), ...
                        Problem.upper(1:Problem.DU));
                else
                    ulEA = zeros(0,Problem.DU);
                end

                % Offspring order: base anchors, same anchors with PPO perturbations, EA offspring.
                ulOffDec = [basePPO;ulPPO;ulEA];

                %% 4. Warm-start lower-level search
                AllDec = Population.decs;
                AllUL  = AllDec(:,1:Problem.DU);
                [~,closest] = min(pdist2(ulOffDec,AllUL),[],2);

                llOffDec = zeros(size(ulOffDec,1),Problem.DL);
                for i = 1 : size(ulOffDec,1)
                    [llOffDec(i,:),totalFElower] = llSearch( ...
                        Problem, ...
                        ulOffDec(i,:), ...
                        AllDec(closest(i),Problem.DU+1:end), ...
                        totalFElower);
                end

                %% 5. Evaluate offspring
                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);

                %% 6. Environmental selection
                [Population,~,~] = EnvironmentalSelectionUpper( ...
                    Problem,Population,Offspring,Nbase,Nppo);

                %% 7. Paired PPO reward: same elite sample with and without perturbation
                [rewardsPPO,~] = RewardCalculator( ...
                    Problem,Offspring,Nbase,Nppo,deltaPPO,eliteInfo.perturbScale);

                %% 8. Record best upper and lower information
                Fitness = CalFitness(Problem.C,Population);
                [UpperFit,best] = min(Fitness);

                bestDec = Population(best).dec;
                bestObj = Population(best).obj;
                bestCon = Population(best).con;

                if length(bestObj) >= 2
                    FL = bestObj(2);
                    xu1 = bestDec(1:Problem.p);
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
                        case 'SMD11'
                            FLstar = sum(xu1.^2,2) + 1;
                        case 'SMD12'
                            FLstar = sum(xu1.^2,2) + 1;
                        otherwise
                            FLstar = nan;
                    end

                    if isnan(FLstar)
                        lowerGap = inf;
                    else
                        lowerGap = abs(FL - FLstar);
                    end
                else
                    FL = nan;
                    lowerGap = inf;
                end

                if isempty(bestCon) || length(bestCon) < Problem.C + 1
                    LLCV = 0;
                else
                    llCon = bestCon(Problem.C+1:end);
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

                fprintf('Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e\n', ...
                    gen, Problem.FE, totalFElower, UpperFit, LowerFit, lowerGap);

                fprintf(fid,'%d\t%d\t%d\t%.12e\t%.12e\t%.12e\n', ...
                    gen, Problem.FE, totalFElower, UpperFit, LowerFit, lowerGap);

                drawnow;

                %% 9. Save one transition per PPO-perturbed offspring
                oneBuffer = InitializePPOBuffer();
                oneBuffer.states       = statesPPO;
                oneBuffer.actions      = actionsPPO;
                oneBuffer.rewards      = rewardsPPO;
                oneBuffer.next_states  = statesPPO;
                oneBuffer.log_probs    = logProbsPPO;
                oneBuffer.values       = valuesPPO;
                oneBuffer.action_means = actionMeansPPO;
                oneBuffer.action_stds  = actionStdsPPO;
                oneBuffer.dones        = ones(Nppo,1);

                if abs(UpperFit) <= upperTol
                    upperReached = true;
                    fprintf('Upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        UpperFit, Problem.FE, totalFElower);
                end

                rolloutBuffer = AppendPPOBuffer(rolloutBuffer,oneBuffer);
                rolloutGenCount = rolloutGenCount + 1;

                %% 10. PPO update
                if upperReached || rolloutGenCount >= rolloutUpdateGap
                    [Actor,Critic] = PPOUpdate(Actor,Critic,rolloutBuffer);
                    rolloutBuffer = InitializePPOBuffer();
                    rolloutGenCount = 0;
                end

                %% 11. Update generation counter
                gen = gen + 1;
            end

            if rolloutGenCount > 0 && ~isempty(rolloutBuffer.states)
                rolloutBuffer.dones(end) = 1;
                [Actor,Critic] = PPOUpdate(Actor,Critic,rolloutBuffer);
            end
        end
    end
end

function CloseLogFile(fid)
    if ~isempty(fid) && isnumeric(fid) && fid > 0
        fclose(fid);
    end
end
