classdef BLRLEA4FinalStats < handle
% Mutable final counters for PlatEMO termination cleanup.

    properties
        UpperFE
        TotalLowerFE
    end

    methods
        function obj = BLRLEA4FinalStats(upperFE,totalLowerFE)
            obj.UpperFE = upperFE;
            obj.TotalLowerFE = totalLowerFE;
        end
    end
end
