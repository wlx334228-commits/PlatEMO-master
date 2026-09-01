function Archive = InitializeTransitionArchive(Problem,maxSize)
% Initialize constraint-aware local transition memories.

    Archive.Feasibility = EmptyTransitionSubArchive(Problem,maxSize);
    Archive.Balanced = EmptyTransitionSubArchive(Problem,maxSize);
    Archive.Objective = EmptyTransitionSubArchive(Problem,maxSize);
    Archive.Count = 0;
end

function SubArchive = EmptyTransitionSubArchive(Problem,maxSize)
    SubArchive.XSource = zeros(0,Problem.DU);
    SubArchive.XTarget = zeros(0,Problem.DU);
    SubArchive.X0 = zeros(0,Problem.DU);
    SubArchive.D = zeros(0,Problem.DU);
    SubArchive.DPair = zeros(0,1);
    SubArchive.DeltaF = zeros(0,1);
    SubArchive.DeltaCV = zeros(0,1);
    SubArchive.I = zeros(0,1);
    SubArchive.W = zeros(0,1);
    SubArchive.Gen = zeros(0,1);
    SubArchive.MaxSize = maxSize;
end
