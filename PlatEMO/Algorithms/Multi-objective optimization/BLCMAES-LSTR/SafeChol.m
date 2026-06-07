function cholC = SafeChol(C)
% Cholesky factor with a small eigenvalue repair fallback.

    C = (C + C')/2;
    [cholC,flag] = chol(C);
    if flag == 0
        return;
    end

    C = RepairCovariance(C);
    [cholC,flag] = chol(C);
    if flag ~= 0
        cholC = eye(size(C,1));
    end
end
