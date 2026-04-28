function [functionValue,equalityConstrVals,inequalityConstrVals] = ulevaluate(ulPop,llPop,Problem)

    noOfMembers = size(ulPop,1);

    functionValue = zeros(noOfMembers,1);
    equalityConstrVals = [];
    inequalityConstrVals = [];

    for i = 1:noOfMembers
        Population = Problem.Evaluation([ulPop(i,:), llPop(i,:)]);
        PopObj = Population.objs;
        PopCon = Population.cons;

        % 上层单目标
        functionValue(i,1) = PopObj(1);

        % 不考虑等式约束
        equalityConstrValsTemp = [];

        % 直接把约束矩阵当作不等式约束
        inequalityConstrValsTemp = PopCon;

        if ~isempty(inequalityConstrValsTemp)
            inequalityConstrVals(i,1:size(inequalityConstrValsTemp,2)) = inequalityConstrValsTemp;
        end
    end
end
