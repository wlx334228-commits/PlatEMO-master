function Archive = InitializeTransitionArchive(Problem,maxSize)
% Initialize constraint-aware local transition memories.

    Archive.Feasibility = EmptyTransitionSubArchive(Problem,maxSize);
    Archive.Objective = EmptyTransitionSubArchive(Problem,maxSize);
    Archive.Count = 0;
end

function SubArchive = EmptyTransitionSubArchive(Problem,maxSize)
    SubArchive.X0 = zeros(0,Problem.DU);
    SubArchive.D = zeros(0,Problem.DU);
    SubArchive.W = zeros(0,1);
    SubArchive.Gen = zeros(0,1);
    SubArchive.MaxSize = maxSize;
end
