function [deltaFinal,sigmaClipped] = ApplySigmaSafetyCap(deltaRaw,CMA,Problem,params)
% Optionally cap an LSTR correction length without changing its direction.

    deltaFinal = deltaRaw;
    sigmaClipped = false;

    if any(~isfinite(deltaRaw))
        deltaFinal = zeros(size(deltaRaw));
        sigmaClipped = true;
        return;
    end
    if ~isfield(params,'enableSigmaCap') || ~params.enableSigmaCap
        return;
    end
    if isempty(CMA) || ~isfield(CMA,'sigma') || CMA.sigma <= 0
        return;
    end

    DU = Problem.DU;
    range = Problem.upper(1:DU) - Problem.lower(1:DU);
    range(range < 1e-12) = 1;
    normalizedDelta = deltaRaw ./ range;
    normalizedNorm = norm(normalizedDelta);
    if normalizedNorm <= 0
        return;
    end

    sigmaNorm = CMA.sigma / max(median(range),eps);
    limit = max(params.kappaSigmaCap,0) * max(sigmaNorm,eps);
    if normalizedNorm > limit
        deltaFinal = deltaRaw .* (limit / normalizedNorm);
        sigmaClipped = true;
    end
end
