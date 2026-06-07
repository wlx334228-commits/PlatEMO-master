function Dec = RepairBounds(Dec,lower,upper)
% Clamp decisions into the bounded search space.

    lower = lower(:)';
    upper = upper(:)';
    Dec = min(max(Dec,repmat(lower,size(Dec,1),1)),repmat(upper,size(Dec,1),1));
end
