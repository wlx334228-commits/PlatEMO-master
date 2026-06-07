function CMA = RepairCMAES(CMA)
% Repair covariance eigensystem and inverse square root.

    CMA.C = (CMA.C + CMA.C')/2;
    [CMA.B,E] = eig(CMA.C);
    eigVal = real(diag(E))';
    if any(eigVal <= 0)
        eigVal(eigVal < 0) = 0;
        eigVal = eigVal + max(max(eigVal)/1e7,1e-12);
        CMA.C = CMA.C + max(max(eigVal)/1e7,1e-12)*eye(CMA.dim);
    end
    eigVal = max(eigVal,1e-12);
    if max(eigVal) > 1e14 * min(eigVal)
        eigVal = max(eigVal,max(eigVal)/1e14);
        CMA.C = CMA.B*diag(eigVal)*CMA.B';
        CMA.C = (CMA.C + CMA.C')/2;
    end
    [CMA.B,E] = eig(CMA.C);
    eigVal = max(real(diag(E))',1e-12);
    CMA.eigD = sqrt(eigVal);
    CMA.invsqrtC = CMA.B * diag(CMA.eigD.^-1) * CMA.B';
end
