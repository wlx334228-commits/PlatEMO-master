classdef BLRLEA2_EAOnly < ALGORITHM
    % <2024> <multi> <real> <constrained/none> <bilevel>
    methods
        function main(Algorithm,Problem)

            %% Total lower-level function evaluations
            totalFElower = 0;

            %% Generate random upper-level population
            ulPopDec = unifrnd( ...
                repmat(Problem.lower(1:Problem.DU),Problem.N,1), ...
                repmat(Problem.upper(1:Problem.DU),Problem.N,1));

            %% Lower-level search for initial population
            llPopDec = zeros(Problem.N,Problem.DL);

            for i = 1 : size(ulPopDec,1)
                [llPopDec(i,:),totalFElower] = llSearch( ...
                    Problem, ...
                    ulPopDec(i,:), ...
                    [], ...
                    totalFElower);
            end

            Population = Problem.Evaluation([ulPopDec,llPopDec]);

            %% Debug log
            gen = 1;
            fid = fopen('BLRLEA2_EAOnly_debug_log.txt','w');
            fprintf(fid,'Gen\tUpperFE\tTotalLowerFE\tUpperFit\tLowerFit\tlowerGap\n');

            %% Upper-level optimization
            while Algorithm.NotTerminated(Population)

                %% 1. Calculate upper-level fitness
                Fitness = CalFitness(Problem.C,Population);

                %% 2. Generate all upper-level offspring by SBX + polynomial mutation
                Noff = Problem.N;

                MatingPool = TournamentSelection(2,Noff,Fitness);
                ParentDec  = Population(MatingPool).decs;

                ulOffDec = OperatorSBXPM( ...
                    ParentDec(:,1:Problem.DU), ...
                    Problem.lower(1:Problem.DU), ...
                    Problem.upper(1:Problem.DU));

                %% 3. Lower-level search
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

                %% 4. Evaluate complete offspring
                Offspring = Problem.Evaluation([ulOffDec,llOffDec]);

                %% 5. Upper-level environmental selection
                Population = EnvironmentalSelectionUpper(Problem,Population,Offspring);

                %% 6. Record best information
                Fitness = CalFitness(Problem.C,Population);
                [UpperFit,best] = min(Fitness);

                bestDec = Population(best).dec;
                bestObj = Population(best).obj;
                bestCon = Population(best).con;

                [LowerFit,lowerGap] = CalOneLowerFitness(Problem,bestDec,bestObj,bestCon);

                fprintf('EAOnly Gen=%4d | UpperFE=%6d | TotalLowerFE=%10d | UpperFit=%.6e | LowerFit=%.6e | lowerGap=%.6e\n', ...
                    gen, Problem.FE, totalFElower, UpperFit, LowerFit, lowerGap);

                fprintf(fid,'%d\t%d\t%d\t%.12e\t%.12e\t%.12e\n', ...
                    gen, Problem.FE, totalFElower, UpperFit, LowerFit, lowerGap);

                drawnow;

                %% 7. Upper-level termination by upper fitness
                upperTol = 1e-6;

                if UpperFit <= upperTol
                    fprintf('EAOnly upper optimum reached: UpperFit = %.6e, UpperFE = %d, TotalLowerFE = %d\n', ...
                        UpperFit, Problem.FE, totalFElower);
                    break;
                end

                gen = gen + 1;
            end

            fclose(fid);
        end
    end
end


function [LowerFit,lowerGap] = CalOneLowerFitness(Problem,Dec,Obj,Con)
% Calculate lower-level fitness and lower-level gap for one complete solution.
%
% LowerFit:
%   If lower-level constraints are feasible:
%       LowerFit = FL
%   Otherwise:
%       LowerFit = LLCV + 1e10
%
% lowerGap:
%   lowerGap = abs(FL - FLstar(xu))

    %% Lower-level objective value
    if length(Obj) >= 2
        FL = Obj(2);
    else
        FL = nan;
    end

    %% lowerGap = abs(FL - FLstar)
    lowerGap = CalOneLowerGap(Problem,Dec,Obj);

    %% Lower-level constraint violation
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

    %% LowerFit follows lower-level fitness logic
    if LLCV <= 0
        LowerFit = FL;
    else
        LowerFit = LLCV + 1e10;
    end
end


function lowerGap = CalOneLowerGap(Problem,Dec,Obj)
% lowerGap = abs(FL - FLstar(xu))

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