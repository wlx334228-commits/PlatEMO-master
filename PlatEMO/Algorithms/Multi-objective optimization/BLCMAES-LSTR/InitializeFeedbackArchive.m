function FeedbackArchive = InitializeFeedbackArchive(Problem,maxSize)
% Initialize weak feedback memory for individual LSTR reliability.

    FeedbackArchive.X = zeros(0,Problem.DU);
    FeedbackArchive.ModeCode = zeros(0,1);
    FeedbackArchive.Usable = false(0,1);
    FeedbackArchive.Applied = false(0,1);
    FeedbackArchive.Success = false(0,1);
    FeedbackArchive.M = zeros(0,1);
    FeedbackArchive.MeanDist = zeros(0,1);
    FeedbackArchive.Rdist = zeros(0,1);
    FeedbackArchive.Rdir = zeros(0,1);
    FeedbackArchive.Rnum = zeros(0,1);
    FeedbackArchive.R = zeros(0,1);
    FeedbackArchive.Lambda = zeros(0,1);
    FeedbackArchive.DRawNorm = zeros(0,1);
    FeedbackArchive.DeltaRawNorm = zeros(0,1);
    FeedbackArchive.DeltaFinalNorm = zeros(0,1);
    FeedbackArchive.SigmaClipped = false(0,1);
    FeedbackArchive.Gen = zeros(0,1);
    FeedbackArchive.Count = 0;
    FeedbackArchive.MaxSize = maxSize;
end
