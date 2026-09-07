classdef BLCMAESLSTR < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    % PlatEMO implementation of BL-CMA-ES with local successful transition reuse.
    methods
        function main(Algorithm,Problem)
            addpath(fileparts(mfilename('fullpath')));

            %% Parameters
            [uTol,lTol,~,~,includeLLConInUpper,archiveScale, ...
                neighborK,legacyReuseThreshold,directionNoise,maxStepRatio, ...
                cvTol,objTol,frLow,frHigh,minNear,minConsistency, ...
                stepSigmaRatio,alphaFeasibility,alphaBalanced, ...
                alphaObjective,enableIndividualReliability,tauR, ...
                wd,wc,wh,betaReliability,feedbackK,feedbackPriorA, ...
                feedbackPriorB,enableSigmaCap,kappaSigmaCap, ...
                historyWindow,archiveThreshold,reuseThreshold, ...
                minHistoricalImprovement,Kuse,Mref,debugRecord] = ...
                Algorithm.ParameterSet(1e-6,1e-6,350,25,0,10,5, ...
                0.06,0.00,0.20, ...
                1e-8,1e-12,0.20,0.70,1,0.30,0.50,0.70,0.50,0.25, ...
                1,0.25,0.30,0.35,0.35,0,8,1,1,0,1.50, ...
                4,0.10,0.06,1e-8,5,3,1);

            BI = BuildBI(Problem,uTol,lTol,includeLLConInUpper);
            CMA = InitCMAES(BI);
            Archive = InitializeTransitionArchive(Problem,max(1,round(archiveScale*Problem.N)));
            FeedbackArchive = InitializeFeedbackArchive(Problem,max(1,round(archiveScale*Problem.N)));
            HistoryWindow = InitializeEvaluatedHistoryWindow(Problem,historyWindow);
            lstrParams.neighborK = neighborK;
            lstrParams.legacyReuseThreshold = legacyReuseThreshold;
            lstrParams.reuseThreshold = reuseThreshold;
            lstrParams.historyWindow = max(1,round(historyWindow));
            lstrParams.archiveThreshold = archiveThreshold;
            lstrParams.archiveRadius = archiveThreshold;
            lstrParams.reuseRadius = reuseThreshold;
            lstrParams.minHistoricalImprovement = minHistoricalImprovement;
            lstrParams.Kuse = Kuse;
            lstrParams.Mref = Mref;
            lstrParams.directionNoise = directionNoise;
            lstrParams.maxStepRatio = maxStepRatio;
            lstrParams.cvTol = cvTol;
            lstrParams.objTol = objTol;
            lstrParams.frLow = frLow;
            lstrParams.frHigh = frHigh;
            lstrParams.minNear = minNear;
            lstrParams.minConsistency = minConsistency;
            lstrParams.stepSigmaRatio = stepSigmaRatio;
            lstrParams.alphaFeasibility = alphaFeasibility;
            lstrParams.alphaBalanced = alphaBalanced;
            lstrParams.alphaObjective = alphaObjective;
            lstrParams.enableIndividualReliability = logical(enableIndividualReliability);
            lstrParams.tauR = tauR;
            lstrParams.reliabilityThreshold = tauR;
            lstrParams.wd = wd;
            lstrParams.wc = wc;
            lstrParams.wh = wh;
            lstrParams.betaReliability = betaReliability;
            lstrParams.feedbackK = feedbackK;
            lstrParams.feedbackPriorA = feedbackPriorA;
            lstrParams.feedbackPriorB = feedbackPriorB;
            lstrParams.enableSigmaCap = logical(enableSigmaCap);
            lstrParams.kappaSigmaCap = kappaSigmaCap;
            lstrParams.debugRecord = logical(debugRecord);

            maxIter = ceil(BI.UmaxFEs/CMA.lambda);

            elite = [];
            feasibleRate = 0;
            totalFElower = 0;
            upperReached = false;
            runRecordWritten = false;
            gen = 0;
            upperRecord = [];
            upperRecordFE = [];
            upperImprIter = max(1,ceil(BI.UmaxImprFEs/CMA.lambda));

            try
                for iter = 1 : maxIter
                    gen = iter;
                    if Problem.FE >= BI.UmaxFEs
                        break;
                    end

                    %% Sampling and LSTR correction
                    popN = min(CMA.lambda,BI.UmaxFEs-Problem.FE);
                    sampledDec = zeros(popN,BI.dim);
                    ulBaseDec = zeros(popN,BI.u_dim);
                    for i = 1 : popN
                        sampledDec(i,:) = SampleFullVector(CMA,BI);
                        ulBaseDec(i,:) = sampledDec(i,1:BI.u_dim);
                    end
                    lstrMode = SelectLSTRMode(feasibleRate,lstrParams);
                    [ulDec,lstrStats,lstrLog] = GenerateLSTROffspring(ulBaseDec,Archive, ...
                        FeedbackArchive,Problem,lstrParams,CMA,lstrMode);

                    POP = EmptyIndividual(popN);
                    for i = 1 : popN
                        POP(i).UX = ulDec(i,:);
                        POP(i).LX = sampledDec(i,BI.u_dim+1:end);
                    end

                    %% Lower-level search and upper-level evaluation
                    for i = 1 : popN
                        [POP(i).LX,POP(i).LF,POP(i).LC,POP(i).RF,totalFElower] = ...
                            LowerLevelSearch(Problem,POP(i).UX,CMA,BI,totalFElower);
                        [POP(i).UF,POP(i).UC,POP(i).Solution] = ...
                            EvaluateUpper(Problem,POP(i).UX,POP(i).LX,BI);
                        POP(i).UFEs = Problem.FE;
                        POP(i).LFEs = totalFElower;
                    end

                    POP = AssignUpperFitness(POP,BI);

                    %% Elite preservation and refinement
                    rfIdx = find([POP.RF]);
                    if isempty(rfIdx)
                        rfIdx = 1 : length(POP);
                    end
                    [~,bestLocal] = min([POP(rfIdx).fit]);
                    bestIdx = rfIdx(bestLocal);
                    bestIndv = POP(bestIdx);

                    if UpperLevelComparator(bestIndv,elite,BI) || rand > 0.5
                        [bestIndv,totalFElower] = Refine(Problem,bestIndv,CMA,BI,totalFElower);
                        POP(bestIdx) = bestIndv;
                        if UpperLevelComparator(bestIndv,elite,BI)
                            elite = bestIndv;
                        end
                    elseif ~isempty(elite)
                        [elite,totalFElower] = Refine(Problem,elite,CMA,BI,totalFElower);
                    end

                    POP = AssignUpperFitness(POP,BI);
                    if isempty(elite)
                        [~,bestIdx] = min([POP.fit]);
                        elite = POP(bestIdx);
                    end
                    selectedIdx = SelectedParentIndices(POP,CMA.mu);
                    if lstrParams.enableIndividualReliability
                        FeedbackArchive = UpdateFeedbackArchive(FeedbackArchive,lstrLog,selectedIdx,iter,lstrParams);
                    end

                    CurrentRecords = PopulationToHistoryRecords(POP,BI,lstrParams.cvTol,iter);
                    [Archive,newKnowledgeN] = UpdateTransitionArchive(Problem,Archive,HistoryWindow, ...
                        CurrentRecords,BI,lstrParams,lstrMode,iter);
                    HistoryWindow = UpdateEvaluatedHistoryWindow(HistoryWindow,CurrentRecords,lstrParams.historyWindow);
                    historyWindowN = size(HistoryWindow.X,1);
                    currentPopulation = [POP.Solution];
                    feasibleRate = CurrentFeasibilityRate(POP,BI,lstrParams.cvTol);

                    elite.UFEs = Problem.FE;
                    elite.LFEs = totalFElower;

                    %% Termination check
                    [lowerFit,lowerGap] = CalOneLowerFitness(Problem,elite);
                    targetBest = isfinite(BI.u_fopt) && abs(bestIndv.UF - BI.u_fopt) < BI.u_ftol;
                    targetElite = isfinite(BI.u_fopt) && abs(elite.UF - BI.u_fopt) < BI.u_ftol;
                    upperAcc = abs(elite.UF - BI.u_fopt);
                    lowerAcc = abs(lowerFit - BI.l_fopt);
                    reachMaxFEs = Problem.FE >= BI.UmaxFEs;
                    upperReached = targetBest || targetElite;
                    upperRecord(end+1) = elite.UF; %#ok<AGROW>
                    upperRecordFE(end+1) = Problem.FE; %#ok<AGROW>
                    upperWindowReached = HasRecentObjectiveRangeBelowTol(upperRecordFE, ...
                        upperRecord,BI.UmaxImprFEs,BI.u_ftol,upperImprIter);

                    [feasArchiveN,balancedArchiveN,objArchiveN] = TransitionArchiveCounts(Archive);
                    feedbackN = FeedbackArchiveSize(FeedbackArchive);
                    fprintf(['BLCMAESLSTR Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | ', ...
                        'UpperFit=%.6e | UpperOpt=%.6e | UAcc=%.6e | ', ...
                        'LowerFit=%.6e | LowerOpt=%.6e | LAcc=%.6e | ', ...
                        'lowerGap=%.6e | RF=%d | FR=%.2f | Mode=%s | ', ...
                        'LSTR=%d/%d | R=%.2f/%.2f | Lambda0=%d | Clip=%d | ', ...
                        'Archive(F/B/O/FB)=%d/%d/%d/%d | Win=%d | NewK=%d | Sigma=%.3e\n'], ...
                        iter,Problem.FE,totalFElower,elite.UF,BI.u_fopt,upperAcc, ...
                        lowerFit,BI.l_fopt,lowerAcc,lowerGap, ...
                        elite.RF,feasibleRate,lstrMode,lstrStats.accepted,lstrStats.total, ...
                        lstrStats.meanR,lstrStats.medianR,lstrStats.zeroLambda, ...
                        lstrStats.sigmaClipped,feasArchiveN,balancedArchiveN,objArchiveN, ...
                        feedbackN,historyWindowN,newKnowledgeN,CMA.sigma);
                    if lstrParams.debugRecord
                        AppendBLCMAESLSTRDebugRecord(Problem,iter,Problem.FE,totalFElower, ...
                            lstrMode,feasibleRate,historyWindowN,newKnowledgeN,Archive, ...
                            FeedbackArchive,lstrStats);
                        AppendBLCMAESLSTRIndividualDebugRecord(Problem,iter,lstrMode,lstrLog);
                    end

                    nofinish = Algorithm.NotTerminated(currentPopulation);

                    if targetBest
                        elite = bestIndv;
                    end
                    if upperWindowReached || upperReached || reachMaxFEs || ~nofinish
                        break;
                    end

                    CMA = UpdateCMAESFromPOP(CMA,POP,BI);
                end
            catch err
                if strcmp(err.identifier,'PlatEMO:Termination')
                    writeRunRecord();
                end
                rethrow(err);
            end

            writeRunRecord();

            function writeRunRecord()
                if ~runRecordWritten
                    AppendBLCMAESLSTRRunFERecord(Problem,Problem.FE,totalFElower,gen,upperReached);
                    runRecordWritten = true;
                end
            end
        end
    end
end

function BI = BuildBI(Problem,uTol,lTol,includeLLConInUpper)
    BI.dim = Problem.D;
    BI.u_dim = Problem.DU;
    BI.l_dim = Problem.DL;
    BI.xrange = [Problem.lower;Problem.upper];
    BI.u_lb = Problem.lower(1:Problem.DU);
    BI.u_ub = Problem.upper(1:Problem.DU);
    BI.l_lb = Problem.lower(Problem.DU+1:end);
    BI.l_ub = Problem.upper(Problem.DU+1:end);
    [UmaxFEs,LmaxFEs,UmaxImprFEs,LmaxImprFEs] = OriginalBLCMAESBudgets(Problem);
    BI.UmaxFEs = min(Problem.maxFE,UmaxFEs);
    BI.LmaxFEs = min(Problem.maxFElower,LmaxFEs);
    BI.UmaxImprFEs = UmaxImprFEs;
    BI.LmaxImprFEs = LmaxImprFEs;
    BI.u_ftol = uTol;
    BI.l_ftol = lTol;
    [BI.u_fopt,BI.l_fopt] = OriginalBLCMAESOptima(Problem);
    BI.upperConN = Problem.C;
    BI.isLowerLevelConstraintsIncludedInUpperLevel = logical(includeLLConInUpper);
end

function mode = SelectLSTRMode(feasibleRate,params)
    if feasibleRate < params.frLow
        mode = 'feasibility';
    elseif feasibleRate < params.frHigh
        mode = 'balanced';
    else
        mode = 'objective';
    end
end

function feasibleRate = CurrentFeasibilityRate(POP,BI,cvTol)
    if isempty(POP)
        feasibleRate = 0;
        return;
    end

    CV = [POP.UC];
    if ~BI.isLowerLevelConstraintsIncludedInUpperLevel
        CV = CV + [POP.LC];
    end
    feasibleRate = sum(CV <= cvTol) / numel(CV);
end

function HistoryWindow = InitializeEvaluatedHistoryWindow(Problem,maxGenerations)
    HistoryWindow.X = zeros(0,Problem.DU);
    HistoryWindow.F = zeros(0,1);
    HistoryWindow.CV = zeros(0,1);
    HistoryWindow.Fit = zeros(0,1);
    HistoryWindow.Feasible = false(0,1);
    HistoryWindow.RF = false(0,1);
    HistoryWindow.Gen = zeros(0,1);
    HistoryWindow.MaxGenerations = max(1,round(maxGenerations));
end

function Records = PopulationToHistoryRecords(POP,BI,cvTol,gen)
    if isempty(POP)
        Records.X = zeros(0,BI.u_dim);
        Records.F = zeros(0,1);
        Records.CV = zeros(0,1);
        Records.Fit = zeros(0,1);
        Records.Feasible = false(0,1);
        Records.RF = false(0,1);
        Records.Gen = zeros(0,1);
        return;
    end

    Records.X = cat(1,POP.UX);
    Records.F = [POP.UF]';
    CV = [POP.UC]';
    if ~BI.isLowerLevelConstraintsIncludedInUpperLevel
        CV = CV + [POP.LC]';
    end
    Records.CV = CV;
    Records.Fit = [POP.fit]';
    Records.Feasible = CV <= cvTol;
    Records.RF = [POP.RF]';
    Records.Gen = gen * ones(length(POP),1);
end

function HistoryWindow = UpdateEvaluatedHistoryWindow(HistoryWindow,Records,maxGenerations)
    if isempty(Records.X)
        return;
    end

    HistoryWindow.X = [HistoryWindow.X;Records.X];
    HistoryWindow.F = [HistoryWindow.F;Records.F];
    HistoryWindow.CV = [HistoryWindow.CV;Records.CV];
    HistoryWindow.Fit = [HistoryWindow.Fit;Records.Fit];
    HistoryWindow.Feasible = [HistoryWindow.Feasible;Records.Feasible];
    HistoryWindow.RF = [HistoryWindow.RF;Records.RF];
    HistoryWindow.Gen = [HistoryWindow.Gen;Records.Gen];
    HistoryWindow.MaxGenerations = max(1,round(maxGenerations));

    minGen = Records.Gen(end) - HistoryWindow.MaxGenerations + 1;
    keep = HistoryWindow.Gen >= minGen;
    HistoryWindow.X = HistoryWindow.X(keep,:);
    HistoryWindow.F = HistoryWindow.F(keep,:);
    HistoryWindow.CV = HistoryWindow.CV(keep,:);
    HistoryWindow.Fit = HistoryWindow.Fit(keep,:);
    HistoryWindow.Feasible = HistoryWindow.Feasible(keep,:);
    HistoryWindow.RF = HistoryWindow.RF(keep,:);
    HistoryWindow.Gen = HistoryWindow.Gen(keep,:);
end

function [feasibilityN,balancedN,objectiveN] = TransitionArchiveCounts(Archive)
    feasibilityN = size(Archive.Feasibility.X0,1);
    if isfield(Archive,'Balanced')
        balancedN = size(Archive.Balanced.X0,1);
    else
        balancedN = 0;
    end
    objectiveN = size(Archive.Objective.X0,1);
end

function feedbackN = FeedbackArchiveSize(FeedbackArchive)
    feedbackN = size(FeedbackArchive.X,1);
end

function selectedIdx = SelectedParentIndices(POP,mu)
    if isempty(POP)
        selectedIdx = [];
        return;
    end

    [~,rank] = sort([POP.fit],'ascend');
    selectedIdx = rank(1:min(mu,length(rank)));
end

function [UmaxFEs,LmaxFEs,UstopFEs,LstopFEs] = OriginalBLCMAESBudgets(Problem)
    if Problem.D == 20
        UmaxFEs = 5000;
        UstopFEs = 750;
        LmaxFEs = 500;
        LstopFEs = 50;
    elseif Problem.D == 10
        UmaxFEs = 3500;
        UstopFEs = 500;
        LmaxFEs = 350;
        LstopFEs = 35;
    else
        UmaxFEs = 2500;
        UstopFEs = 350;
        LmaxFEs = 250;
        LstopFEs = 25;
    end
end

function [uOpt,lOpt] = OriginalBLCMAESOptima(Problem)
    optDec = OriginalSMDOptimumDecision(Problem);
    if ~isempty(optDec)
        optObj = Problem.CalObj(optDec);
        uOpt = optObj(1);
        lOpt = optObj(2);
        return;
    end

    switch class(Problem)
        case 'TP1'
            uOpt = 225; lOpt = 100;
        case 'TP2'
            uOpt = 0; lOpt = 100;
        case 'TP3'
            uOpt = -18.6787; lOpt = -1.0156;
        case 'TP4'
            uOpt = -29.2; lOpt = 3.2;
        case 'TP5'
            uOpt = -3.6; lOpt = -2;
        case 'TP6'
            uOpt = -1.2098; lOpt = 7.6172;
        case 'TP7'
            uOpt = -1.961; lOpt = 1.961;
        case 'TP8'
            uOpt = 0; lOpt = 100;
        case {'TP9','TP10'}
            uOpt = 0; lOpt = 1;
        otherwise
            uOpt = nan; lOpt = nan;
    end
end

function optDec = OriginalSMDOptimumDecision(Problem)
    problemName = class(Problem);
    if ~strncmp(problemName,'SMD',3)
        optDec = [];
        return;
    end

    r = floor(Problem.DU/2);
    p = Problem.DU - r;
    q = Problem.DL - r;
    xu = zeros(1,Problem.DU);
    xl = zeros(1,Problem.DL);

    switch problemName
        case 'SMD2'
            xl = [zeros(1,q),ones(1,r)];
        case 'SMD5'
            xl = [ones(1,q),zeros(1,r)];
        case 'SMD7'
            xl = [zeros(1,q),ones(1,r)];
        case 'SMD8'
            xl = [ones(1,q),zeros(1,r)];
        case 'SMD10'
            a = 1/sqrt(p+r-1);
            b = 1/sqrt(q-1);
            xu = a*ones(1,Problem.DU);
            xl = [b*ones(1,q),atan(a*ones(1,r))];
        case 'SMD11'
            xl = [zeros(1,q),exp(-1/sqrt(r))*ones(1,r)];
        case 'SMD12'
            a = 1/sqrt(p+r-1);
            b = 1/sqrt(q-1);
            xu = a*ones(1,Problem.DU);
            xl = [b*ones(1,q),atan(a-1/sqrt(r))*ones(1,r)];
    end

    optDec = [xu,xl];
end

function POP = EmptyIndividual(N)
    indiv = struct('UX',[],'LX',[],'UF',[],'LF',[],'UC',0,'LC',0, ...
        'RF',false,'fit',[],'UFEs',[],'LFEs',[],'Solution',[]);
    POP = repmat(indiv,1,N);
end

function U = SampleFullVector(CMA,BI)
    U = CMA.xmean + CMA.sigma * randn(1,BI.dim) .* CMA.D * CMA.B';
    over = U > BI.xrange(2,:);
    U(over) = (CMA.xmean(over) + BI.xrange(2,over))/2;
    under = U < BI.xrange(1,:);
    U(under) = (CMA.xmean(under) + BI.xrange(1,under))/2;
end

function [F,C,Solution] = EvaluateUpper(Problem,UX,LX,BI)
    Solution = Problem.Evaluation([UX,LX]);
    F = Solution.obj(1);
    C = SumConstraintViolation(Solution.con(1:min(BI.upperConN,length(Solution.con))));
end

function [F,C,Solution] = EvaluateLower(Problem,UX,LX,BI)
    Solution = Problem.EvaluationLower([UX,LX]);
    F = Solution.obj(2);
    lowerCon = [];
    if length(Solution.con) > BI.upperConN
        lowerCon = Solution.con(BI.upperConN+1:end);
    end
    C = SumConstraintViolation(lowerCon);
end

function C = SumConstraintViolation(Con)
    if isempty(Con)
        C = 0;
    else
        Con(isnan(Con)) = 0;
        C = sum(max(0,Con));
    end
end

function POP = AssignLowerFitness(POP)
    fit = CombineConstraintWithFitness([POP.LF],[POP.LC]);
    for i = 1 : length(POP)
        POP(i).fit = fit(i);
    end
end

function POP = AssignUpperFitness(POP,BI)
    CV = [POP.UC];
    if ~BI.isLowerLevelConstraintsIncludedInUpperLevel
        CV = CV + [POP.LC];
    end
    fit = CombineConstraintWithFitness([POP.UF],CV);
    for i = 1 : length(POP)
        POP(i).fit = fit(i);
    end
end

function fitness = CombineConstraintWithFitness(obj,cv)
    cv = max(0,cv);
    feasible = cv <= 0;
    fitness = obj;
    if any(~feasible)
        if any(feasible)
            base = max(obj(feasible));
        else
            base = 1e10;
        end
        fitness(~feasible) = base + cv(~feasible);
    end
end

function [Q,totalFElower] = Refine(Problem,P,CMA,BI,totalFElower)
    Q = P;
    [Q.LX,Q.LF,Q.LC,Q.RF,totalFElower] = LowerLevelSearch(Problem,Q.UX,CMA,BI,totalFElower);
    if LowerLevelComparator(Q,P)
        Q.RF = max(Q.RF,P.RF);
        [Q.UF,Q.UC,Q.Solution] = EvaluateUpper(Problem,Q.UX,Q.LX,BI);
    else
        Q = P;
    end
end

function noWorse = UpperLevelComparator(P,Q,BI)
    if isempty(Q)
        noWorse = true;
    else
        tmp = AssignUpperFitness([P,Q],BI);
        noWorse = tmp(1).fit <= tmp(2).fit;
    end
end

function noWorse = LowerLevelComparator(P,Q)
    if isempty(Q)
        noWorse = true;
    else
        tmp = AssignLowerFitness([P,Q]);
        noWorse = tmp(1).fit <= tmp(2).fit;
    end
end

function [bestLX,bestLF,bestLC,bestRF,totalFElower] = LowerLevelSearch(Problem,xu,CMA,BI,totalFElower)
    sigma0 = 1;
    LCMA.xmean = CMA.xmean(BI.u_dim+1:end);
    LCMA.sigma = sigma0;
    LCMA.C = CMA.C(BI.u_dim+1:end,BI.u_dim+1:end) * CMA.sigma^2;
    LCMA.pc = CMA.pc(BI.u_dim+1:end) * CMA.sigma;
    LCMA.ps = zeros(1,BI.l_dim);
    lambda = 4 + floor(3*log(BI.l_dim));
    mu = floor(lambda/2);
    weights = log(mu+1/2) - log(1:mu);
    weights = weights/sum(weights);
    mueff = sum(weights)^2/sum(weights.^2);
    cc = (4+mueff/BI.l_dim) / (BI.l_dim+4 + 2*mueff/BI.l_dim);
    cs = (mueff+2) / (BI.l_dim+mueff+5);
    c1 = 2 / ((BI.l_dim+1.3)^2+mueff);
    cmu = min(1-c1, 2*(mueff-2+1/mueff) / ((BI.l_dim+2)^2+mueff));
    damps = 1 + 2*max(0, sqrt((mueff-1)/(BI.l_dim+1))-1) + cs;
    chiN = BI.l_dim^0.5*(1-1/(4*BI.l_dim)+1/(21*BI.l_dim^2));
    [LCMA.B,LCMA.D] = eig(LCMA.C);
    LCMA.D = sqrt(max(diag(LCMA.D),eps))';
    LCMA = RepairCMA(LCMA);
    LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';
    cy = sqrt(BI.l_dim) + 2*BI.l_dim/(BI.l_dim+2);

    bestIndv = [];
    bestRF = false;
    imprIter = max(1,ceil(BI.LmaxImprFEs/lambda));
    record = [];
    recordFE = [];
    localFE = 0;

    while localFE < BI.LmaxFEs
        popN = min(lambda,BI.LmaxFEs-localFE);
        Q = EmptyLowerIndividual(popN);
        for i = 1 : popN
            Q(i).LX = LCMA.xmean + LCMA.sigma * randn(1,BI.l_dim) .* LCMA.D * LCMA.B';
            over = Q(i).LX > BI.l_ub;
            Q(i).LX(over) = (LCMA.xmean(over) + BI.l_ub(over))/2;
            under = Q(i).LX < BI.l_lb;
            Q(i).LX(under) = (LCMA.xmean(under) + BI.l_lb(under))/2;
            [Q(i).LF,Q(i).LC] = EvaluateLower(Problem,xu,Q(i).LX,BI);
            totalFElower = totalFElower + 1;
            localFE = localFE + 1;
        end

        Q = AssignLowerFitness(Q);
        [~,rank] = sort([Q.fit],'ascend');
        xold = LCMA.xmean;
        X = cat(1,Q.LX);
        useMu = min(mu,length(rank));
        curWeights = weights(1:useMu);
        curWeights = curWeights/sum(curWeights);
        curMueff = sum(curWeights)^2/sum(curWeights.^2);
        Y = bsxfun(@minus,X(rank(1:useMu),:),xold) / LCMA.sigma;
        Y = bsxfun(@times,Y,min(1,cy./sqrt(sum((Y*LCMA.invsqrtC').^2,2))));
        deltaXmean = curWeights * Y;
        LCMA.xmean = LCMA.xmean + deltaXmean * LCMA.sigma;
        Cmu = Y' * diag(curWeights) * Y;
        LCMA.ps = (1-cs)*LCMA.ps + sqrt(cs*(2-cs)*curMueff) * deltaXmean * LCMA.invsqrtC;
        LCMA.pc = (1-cc)*LCMA.pc + sqrt(cc*(2-cc)*curMueff) * deltaXmean;
        LCMA.C = (1-c1-cmu) * LCMA.C + c1 * (LCMA.pc'*LCMA.pc) + cmu * Cmu;
        deltaSigma = (cs/damps)*(norm(LCMA.ps)/chiN - 1);
        LCMA.sigma = LCMA.sigma * exp(min(CMA.delta_sigma_max,deltaSigma));
        LCMA.C = triu(LCMA.C) + triu(LCMA.C,1)';
        [LCMA.B,LCMA.D] = eig(LCMA.C);
        LCMA.D = sqrt(max(diag(LCMA.D),eps))';
        LCMA = RepairCMA(LCMA);
        LCMA.invsqrtC = LCMA.B * diag(LCMA.D.^-1) * LCMA.B';

        if LowerLevelComparator(Q(rank(1)),bestIndv)
            bestIndv = Q(rank(1));
        end
        record(end+1) = bestIndv.LF; %#ok<AGROW>
        recordFE(end+1) = localFE; %#ok<AGROW>

        if HasRecentObjectiveRangeBelowTol(recordFE,record,BI.LmaxImprFEs,BI.l_ftol,imprIter)
            bestRF = true;
            break;
        end
    end

    bestLX = bestIndv.LX;
    bestLF = bestIndv.LF;
    bestLC = bestIndv.LC;
end

function Q = EmptyLowerIndividual(N)
    indiv = struct('LX',[],'LF',[],'LC',0,'fit',[]);
    Q = repmat(indiv,1,N);
end

function CMA = InitCMAES(BI)
    CMA.lambda = 4 + floor(3*log(BI.dim));
    CMA.sigma = 0.3 * median(BI.xrange(2,:) - BI.xrange(1,:));
    CMA.mu = floor(CMA.lambda/2);
    CMA.weights = log(CMA.mu+1/2) - log(1:CMA.mu);
    CMA.weights = CMA.weights/sum(CMA.weights);
    CMA.mueff = sum(CMA.weights)^2/sum(CMA.weights.^2);
    CMA.cc = (4+CMA.mueff/BI.dim) / (BI.dim+4 + 2*CMA.mueff/BI.dim);
    CMA.cs = (CMA.mueff+2) / (BI.dim+CMA.mueff+5);
    CMA.c1 = 2 / ((BI.dim+1.3)^2+CMA.mueff);
    CMA.cmu = min(1-CMA.c1, 2*(CMA.mueff-2+1/CMA.mueff) / ((BI.dim+2)^2+CMA.mueff));
    CMA.damps = 1 + 2*max(0, sqrt((CMA.mueff-1)/(BI.dim+1))-1) + CMA.cs;
    CMA.chiN = BI.dim^0.5*(1-1/(4*BI.dim)+1/(21*BI.dim^2));
    CMA.pc = zeros(1,BI.dim);
    CMA.ps = zeros(1,BI.dim);
    CMA.B = eye(BI.dim);
    CMA.D = ones(1,BI.dim);
    CMA.C = CMA.B * diag(CMA.D.^2) * CMA.B';
    CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';
    CMA.xmean = (BI.xrange(2,:) - BI.xrange(1,:)).*rand(1,BI.dim) + BI.xrange(1,:);
    CMA.cy = sqrt(BI.dim) + 2*BI.dim/(BI.dim+2);
    CMA.delta_sigma_max = 1;
end

function CMA = UpdateCMAESFromPOP(CMA,POP,~)
    [~,rank] = sort([POP.fit],'ascend');
    useMu = min(CMA.mu,length(rank));
    weights = CMA.weights(1:useMu);
    weights = weights/sum(weights);
    mueff = sum(weights)^2/sum(weights.^2);
    Selected = [POP(rank(1:useMu)).Solution];
    X = Selected.decs;
    xold = CMA.xmean;
    Y = bsxfun(@minus,X,xold) / CMA.sigma;
    Y = bsxfun(@times,Y,min(1,CMA.cy./sqrt(sum((Y*CMA.invsqrtC').^2,2))));
    deltaXmean = weights * Y;
    CMA.xmean = CMA.xmean + deltaXmean * CMA.sigma;
    Cmu = Y' * diag(weights) * Y;
    CMA.ps = (1-CMA.cs)*CMA.ps + sqrt(CMA.cs*(2-CMA.cs)*mueff) * deltaXmean * CMA.invsqrtC;
    CMA.pc = (1-CMA.cc)*CMA.pc + sqrt(CMA.cc*(2-CMA.cc)*mueff) * deltaXmean;
    CMA.C = (1-CMA.c1-CMA.cmu) * CMA.C + CMA.c1 * (CMA.pc'*CMA.pc) + CMA.cmu * Cmu;
    deltaSigma = (CMA.cs/CMA.damps)*(norm(CMA.ps)/CMA.chiN - 1);
    CMA.sigma = CMA.sigma * exp(min(deltaSigma,CMA.delta_sigma_max));
    CMA.C = triu(CMA.C) + triu(CMA.C,1)';
    [CMA.B,CMA.D] = eig(CMA.C);
    CMA.D = sqrt(max(diag(CMA.D),eps))';
    CMA = RepairCMA(CMA);
    CMA.invsqrtC = CMA.B * diag(CMA.D.^-1) * CMA.B';
end

function Model = RepairCMA(Model)
    dim = length(Model.D);
    if any(Model.D <= 0)
        Model.D(Model.D < 0) = 0;
        tmp = max(Model.D)/1e7;
        if tmp <= 0
            tmp = eps;
        end
        Model.C = Model.C + tmp * eye(dim);
        Model.D = Model.D + tmp * ones(1,dim);
    end
    if max(Model.D) > 1e7 * min(Model.D)
        tmp = max(Model.D)/1e7 - min(Model.D);
        Model.C = Model.C + tmp * eye(dim);
        Model.D = Model.D + tmp * ones(1,dim);
    end
    if Model.sigma > 1e7 * max(Model.D)
        fac = Model.sigma / max(Model.D);
        Model.sigma = Model.sigma / fac;
        Model.D = Model.D * fac;
        Model.pc = Model.pc * fac;
        Model.C = Model.C * fac^2;
    end
end

function reached = HasRecentObjectiveRangeBelowTol(recordFE,recordObj,windowFEs,tol,minRecordN)
    reached = false;
    if length(recordObj) < minRecordN
        return;
    end

    startFE = recordFE(end) - windowFEs + 1;
    idx = recordFE >= startFE;
    if sum(idx) < 2
        return;
    end

    recent = recordObj(idx);
    reached = max(recent) - min(recent) < tol;
end

function [LowerFit,lowerGap] = CalOneLowerFitness(Problem,elite)
    LowerFit = elite.LF;
    lowerGap = CalOneLowerGap(Problem,[elite.UX,elite.LX],elite.LF);
end

function lowerGap = CalOneLowerGap(Problem,Dec,FL)
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
        case 'SMD9'
            FLstar = sum(xu1.^2,2);
        case 'SMD10'
            FLstar = sum(xu1.^2,2) + SMD10LowerOffset(Problem);
        case 'SMD11'
            FLstar = sum(xu1.^2,2) + 1;
        case 'SMD12'
            FLstar = sum(xu1.^2,2) + SMD10LowerOffset(Problem) + 1;
        otherwise
            FLstar = nan;
    end

    if isnan(FLstar)
        lowerGap = inf;
    else
        lowerGap = abs(FL - FLstar);
    end
end

function offset = SMD10LowerOffset(Problem)
    if Problem.q <= 1
        offset = 0;
    else
        xOpt = 1/sqrt(Problem.q-1);
        offset = Problem.q * (xOpt-2)^2;
    end
end
