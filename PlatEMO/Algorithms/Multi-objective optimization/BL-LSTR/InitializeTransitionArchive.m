function Archive = InitializeTransitionArchive(Problem,maxSize)
% Initialize local successful transition memory.

    Archive.X0 = zeros(0,Problem.DU);
    Archive.D = zeros(0,Problem.DU);
    Archive.W = zeros(0,1);
    Archive.Gen = zeros(0,1);
    Archive.MaxSize = maxSize;
    Archive.Count = 0;
end
