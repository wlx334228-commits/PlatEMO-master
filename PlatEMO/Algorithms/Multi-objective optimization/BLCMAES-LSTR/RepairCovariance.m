function C = RepairCovariance(C)
% Keep the covariance matrix symmetric positive semidefinite.

    C = (C + C')/2;
    [V,E] = eig(C);
    eigVal = max(real(diag(E)),1e-12);
    C = real(V*diag(eigVal)*V');
    C = (C + C')/2;
end
