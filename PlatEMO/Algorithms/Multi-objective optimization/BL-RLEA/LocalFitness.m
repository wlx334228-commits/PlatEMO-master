function fitness = LocalFitness(obj, con)
% LocalFitness
% Lower-level fitness with constraint handling

    if isempty(con)
        con = zeros(size(obj,1),1);
    end

    PopCon = sum(max(0, con), 2);
    Feasible = PopCon <= 0;

    fitness = Feasible .* obj + ~Feasible .* (PopCon + 1e10);
end