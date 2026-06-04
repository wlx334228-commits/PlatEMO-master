function [problemName,runID] = PrepareBLRLEA4Run(Problem)
% Prepare problem and run identifiers.

    persistent runCounter
    if isempty(runCounter)
        runCounter = 0;
    end
    runCounter = runCounter + 1;

    problemName = class(Problem);
    t = clock;
    sec = floor(t(6));
    milliSec = floor((t(6) - sec) * 1000);
    runID = sprintf('%04d%02d%02d_%02d%02d%02d_%03d_%03d', ...
        t(1),t(2),t(3),t(4),t(5),sec,milliSec,runCounter);
end
