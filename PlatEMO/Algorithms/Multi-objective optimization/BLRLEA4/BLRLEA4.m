classdef BLRLEA4 < ALGORITHM
    % <2026> <multi> <real> <constrained/none> <bilevel>
    % Actor-generated upper distribution with elite-distribution imitation

    methods
        function main(Algorithm,Problem)
            %% Parameters
            ppoRatio = 0.5;
            stateDim = 2 * Problem.DU;
            actionDim = 2 * Problem.DU;
            hiddenDim = 32;
            learnRate = 1e-3;
            rolloutUpdateGap = 3;
            imitationEpochs = 300;

            %% Networks and buffer
            Actor  = InitializeActorNetwork(stateDim,actionDim,hiddenDim,learnRate);
            Actor.DU = Problem.DU;
            Actor.eliteLossCoef = 0.30;
            Critic = InitializeCriticNetwork(stateDim,hiddenDim,learnRate);

            rolloutBuffer = InitializePPOBuffer();
            rolloutGenCount = 0;

            %% Counters and run record
            totalFElower = 0;
            gen = 1;
            upperTol = 1e-5;
            finalStats = BLRLEA4FinalStats(Problem.FE,totalFElower);
            [problemName,runID] = PrepareBLRLEA4Run(Problem);
            finalCleaner = onCleanup(@()FinalizeBLRLEA4Run(problemName,runID,finalStats)); %#ok<NASGU>

            %% Initial population
            ulPopDec = unifrnd( ...
                repmat(Problem.lower(1:Problem.DU),Problem.N,1), ...
                repmat(Problem.upper(1:Problem.DU),Problem.N,1));

            llPopDec = zeros(Problem.N,Problem.DL);
            for i = 1 : Problem.N
                [llPopDec(i,:),totalFElower] = llSearch(Problem,ulPopDec(i,:),[],totalFElower);
            end
            Population = Problem.Evaluation([ulPopDec,llPopDec]);
            finalStats.UpperFE = Problem.FE;
            finalStats.TotalLowerFE = totalFElower;

            noImproveGen = 0;
            oldBestForState = min(CalFitness(Problem.C,Population));

            %% Upper-level evolution
            while Algorithm.NotTerminated(Population)
                currentFit = CalFitness(Problem.C,Population);
                currentBest = min(currentFit);
                if abs(currentBest) <= upperTol
                    fprintf('BLRLEA4 upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        currentBest,Problem.FE,totalFElower);
                    break;
                end

                OldPopulation = Population;
                oldFit = currentFit;
                oldBest = min(oldFit);
                oldMean = mean(oldFit);

                %% 1. State is the current elite distribution only
                [state,eliteDist] = BuildState(Problem,Population,oldBestForState,noImproveGen);
                Actor = SupervisedActorUpdate(Actor,state,eliteDist,imitationEpochs);
                value = CriticForward(Critic,state);

                %% 2. Actor directly outputs upper distribution parameters
                [action,logProb,actionMean,actionStd] = ActorForward(Actor,state);
                modelDist = ActionToDistribution(action,Problem);

                %% 3. Generate upper offspring: half PPO distribution, half SBXPM
                Noff = Problem.N;
                Nppo = floor(ppoRatio * Noff);
                Nea  = Noff - Nppo;

                ulPPO = GenerateUpperFromDistribution(modelDist,Nppo,Problem);

                if Nea > 0
                    MatingPool = TournamentSelection(2,Nea,oldFit);
                    ParentDec = Population(MatingPool).decs;
                    ulEA = OperatorSBXPM( ...
                        ParentDec(:,1:Problem.DU), ...
                        Problem.lower(1:Problem.DU), ...
                        Problem.upper(1:Problem.DU));
                else
                    ulEA = zeros(0,Problem.DU);
                end

                ulOffDec = [ulPPO;ulEA];

                %% 4. Lower-level search with nearest warm start
                AllDec = Population.decs;
                AllUL  = AllDec(:,1:Problem.DU);
                [~,closest] = min(pdist2(single(ulOffDec),single(AllUL)),[],2);

                llOffDec = zeros(size(ulOffDec,1),Problem.DL);
                for i = 1 : size(ulOffDec,1)
                    [llOffDec(i,:),totalFElower] = llSearch( ...
                        Problem,ulOffDec(i,:), ...
                        AllDec(closest(i),Problem.DU+1:end),totalFElower);
                end

                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);

                %% 5. Environmental selection and PPO-only survival
                [Population,ppoSurvivalRate,ppoSurvived] = EnvironmentalSelectionUpper( ...
                    Problem,Population,Offspring,Nppo,1);

                %% 6. Reward only evaluates PPO distribution and PPO offspring
                rewardStats = RewardCalculator( ...
                    Problem,Offspring(1:Nppo),ppoSurvivalRate,ppoSurvived, ...
                    oldBest,oldMean,eliteDist,modelDist);
                reward = rewardStats.reward;

                %% 7. Build next state and store transition
                newFit = CalFitness(Problem.C,Population);
                [newBest,best] = min(newFit);
                [LowerFit,lowerGap] = CalOneLowerFitness( ...
                    Problem,Population(best).dec,Population(best).obj,Population(best).con);
                if newBest < oldBest - 1e-12
                    nextNoImproveGen = 0;
                else
                    nextNoImproveGen = noImproveGen + 1;
                end

                [nextState,~] = BuildState(Problem,Population,oldBest,nextNoImproveGen);

                oneBuffer = InitializePPOBuffer();
                oneBuffer.states = state;
                oneBuffer.actions = action;
                oneBuffer.rewards = reward;
                oneBuffer.next_states = nextState;
                oneBuffer.log_probs = logProb;
                oneBuffer.values = value;
                oneBuffer.dones = double(abs(newBest) <= upperTol);
                oneBuffer.action_means = actionMean;
                oneBuffer.action_stds = actionStd;
                oneBuffer.elite_mu_targets = eliteDist.muNorm;
                oneBuffer.elite_sigma_targets = eliteDist.sigmaNorm;

                rolloutBuffer = AppendPPOBuffer(rolloutBuffer,oneBuffer);
                rolloutGenCount = rolloutGenCount + 1;

                if rolloutGenCount >= rolloutUpdateGap || oneBuffer.dones
                    [Actor,Critic] = PPOUpdate(Actor,Critic,rolloutBuffer);
                    rolloutBuffer = InitializePPOBuffer();
                    rolloutGenCount = 0;
                end

                finalStats.UpperFE = Problem.FE;
                finalStats.TotalLowerFE = totalFElower;

                fprintf('BLRLEA4 Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e | Reward=%.3e | Match=%.3e | PPOMeanFit=%.3e\n', ...
                    gen,Problem.FE,totalFElower,newBest,LowerFit,lowerGap,reward, ...
                    rewardStats.matchLoss,rewardStats.ppoMeanFit);
                drawnow;

                if abs(newBest) <= upperTol
                    fprintf('BLRLEA4 upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        newBest,Problem.FE,totalFElower);
                    break;
                end

                gen = gen + 1;
                noImproveGen = nextNoImproveGen;
                oldBestForState = oldBest;
            end

            if rolloutGenCount > 0 && ~isempty(rolloutBuffer.states)
                rolloutBuffer.dones(end) = 1;
                [Actor,Critic] = PPOUpdate(Actor,Critic,rolloutBuffer);
            end
        end
    end
end
