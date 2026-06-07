function Dec = RepairBoundsToMean(Dec,xmean,lower,upper)
% Repair out-of-bound samples to the midpoint between mean and boundary.

    xmean = xmean(:)';
    lower = lower(:)';
    upper = upper(:)';
    for j = 1 : size(Dec,2)
        over = Dec(:,j) > upper(j);
        Dec(over,j) = (xmean(j) + upper(j))/2;
        under = Dec(:,j) < lower(j);
        Dec(under,j) = (xmean(j) + lower(j))/2;
    end
end
