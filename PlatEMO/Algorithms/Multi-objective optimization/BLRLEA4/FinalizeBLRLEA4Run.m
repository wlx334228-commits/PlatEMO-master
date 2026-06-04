function FinalizeBLRLEA4Run(problemName,runID,finalStats)
% Append final FE counters when the run finishes or PlatEMO terminates.

    try
        AppendRunFERecord(problemName,runID,finalStats.UpperFE,finalStats.TotalLowerFE);
    catch err
        warning('BLRLEA4:FinalizeFailed','Failed to write final FE record: %s',err.message);
    end
end
