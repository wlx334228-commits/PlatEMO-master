function Fitness = CalFitness(C,Population)
% Calculate single-level fitness for upper or lower populations.

    PopObj = Population.objs;
    PopCon = Population.cons;
    if any(isnan(PopObj(:,1)))
        PopObj = PopObj(:,2);
        PopCon = PopCon(:,C+1:end);
    else
        PopObj = PopObj(:,1);
        PopCon = PopCon(:,1:C);
    end

    if isempty(PopCon)
        PopCon = zeros(size(PopObj,1),1);
    else
        PopCon = sum(max(0,PopCon),2);
    end

    feasible = PopCon <= 0;
    Fitness = feasible.*PopObj + ~feasible.*(PopCon + 1e10);
end
