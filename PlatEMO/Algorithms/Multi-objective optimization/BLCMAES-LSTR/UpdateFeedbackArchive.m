function FeedbackArchive = UpdateFeedbackArchive(FeedbackArchive,genLog,selectedIdx,gen,params)
% Store weak selection feedback for individual LSTR reliability.

    if isempty(genLog)
        return;
    end

    selected = false(1,numel(genLog));
    selectedIdx = selectedIdx(selectedIdx >= 1 & selectedIdx <= numel(genLog));
    selected(selectedIdx) = true;

    for i = 1 : numel(genLog)
        FeedbackArchive.Count = FeedbackArchive.Count + 1;
        FeedbackArchive.X(end+1,:) = genLog(i).X;
        FeedbackArchive.ModeCode(end+1,1) = genLog(i).ModeCode;
        FeedbackArchive.Usable(end+1,1) = genLog(i).Usable;
        FeedbackArchive.Applied(end+1,1) = genLog(i).Applied;
        FeedbackArchive.Success(end+1,1) = selected(i);
        FeedbackArchive.Rdist(end+1,1) = genLog(i).Rdist;
        FeedbackArchive.Rdir(end+1,1) = genLog(i).Rdir;
        FeedbackArchive.Rhist(end+1,1) = genLog(i).Rhist;
        FeedbackArchive.R(end+1,1) = genLog(i).R;
        FeedbackArchive.Lambda(end+1,1) = genLog(i).Lambda;
        FeedbackArchive.DRawNorm(end+1,1) = genLog(i).DRawNorm;
        FeedbackArchive.DeltaRawNorm(end+1,1) = genLog(i).DeltaRawNorm;
        FeedbackArchive.DeltaFinalNorm(end+1,1) = genLog(i).DeltaFinalNorm;
        FeedbackArchive.SigmaClipped(end+1,1) = genLog(i).SigmaClipped;
        FeedbackArchive.Gen(end+1,1) = gen;
    end

    FeedbackArchive = TrimFeedbackArchive(FeedbackArchive,params);
end

function FeedbackArchive = TrimFeedbackArchive(FeedbackArchive,params)
    maxSize = FeedbackArchive.MaxSize;
    if isfield(params,'feedbackMaxSize') && params.feedbackMaxSize > 0
        maxSize = params.feedbackMaxSize;
    end

    if size(FeedbackArchive.X,1) <= maxSize
        return;
    end

    keep = size(FeedbackArchive.X,1) - maxSize + 1 : size(FeedbackArchive.X,1);
    FeedbackArchive.X = FeedbackArchive.X(keep,:);
    FeedbackArchive.ModeCode = FeedbackArchive.ModeCode(keep,:);
    FeedbackArchive.Usable = FeedbackArchive.Usable(keep,:);
    FeedbackArchive.Applied = FeedbackArchive.Applied(keep,:);
    FeedbackArchive.Success = FeedbackArchive.Success(keep,:);
    FeedbackArchive.Rdist = FeedbackArchive.Rdist(keep,:);
    FeedbackArchive.Rdir = FeedbackArchive.Rdir(keep,:);
    FeedbackArchive.Rhist = FeedbackArchive.Rhist(keep,:);
    FeedbackArchive.R = FeedbackArchive.R(keep,:);
    FeedbackArchive.Lambda = FeedbackArchive.Lambda(keep,:);
    FeedbackArchive.DRawNorm = FeedbackArchive.DRawNorm(keep,:);
    FeedbackArchive.DeltaRawNorm = FeedbackArchive.DeltaRawNorm(keep,:);
    FeedbackArchive.DeltaFinalNorm = FeedbackArchive.DeltaFinalNorm(keep,:);
    FeedbackArchive.SigmaClipped = FeedbackArchive.SigmaClipped(keep,:);
    FeedbackArchive.Gen = FeedbackArchive.Gen(keep,:);
end
