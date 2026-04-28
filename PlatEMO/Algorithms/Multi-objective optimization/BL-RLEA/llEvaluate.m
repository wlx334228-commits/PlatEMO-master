function [functionValue, equalityConstrVals, inequalityConstrVals] = llEvaluate(llPop, Problem, ulMember)
% llEvaluate
% Evaluate lower-level objective values and constraints

    noOfMembers = size(llPop,1);

    functionValue = zeros(noOfMembers,1);
    equalityConstrVals = [];
    inequalityConstrVals = [];

    for i = 1:noOfMembers
        if size(ulMember,1) == 1
            dec = [ulMember, llPop(i,:)];
        elseif size(ulMember,1) == noOfMembers
            dec = [ulMember(i,:), llPop(i,:)];
        else
            error('llEvaluate: size mismatch between ulMember and llPop.');
        end

        llPopulation = Problem.EvaluationLower(dec);

        llObj = llPopulation.objs;
        llCon = llPopulation.cons;

        % 下层目标一般在第2列；若只有1列就取第1列
        if size(llObj,2) >= 2
            functionValue(i,1) = llObj(2);
        else
            functionValue(i,1) = llObj(1);
        end

        if ~isempty(llCon)
            llCon = llCon(~isnan(llCon));
            if ~isempty(llCon)
                inequalityConstrVals(i,1:length(llCon)) = llCon;
            end
        end
    end
end