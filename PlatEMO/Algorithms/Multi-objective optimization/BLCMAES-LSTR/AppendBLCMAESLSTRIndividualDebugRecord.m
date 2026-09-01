function AppendBLCMAESLSTRIndividualDebugRecord(Problem,generation,mode,genLog)
% Append individual-level diagnostics for the LSTR reliability gate.

    if isempty(genLog)
        return;
    end

    folder = fileparts(mfilename('fullpath'));
    file = fullfile(folder,'BLCMAESLSTR_lstr_individual_debug.csv');
    writeHeader = ~exist(file,'file');

    fid = fopen(file,'a');
    if fid < 0
        return;
    end

    cleaner = onCleanup(@()fclose(fid));
    if writeHeader
        fprintf(fid,['Problem,D,DU,DL,Generation,Mode,Individual,Usable,Applied,', ...
            'M,MeanDist,Rdist,Rdir,Rnum,R,Lambda,DRawNorm,LambdaDRawNorm,', ...
            'DeltaFinalNorm,SigmaClipped\n']);
    end

    for i = 1 : numel(genLog)
        fprintf(fid,['%s,%d,%d,%d,%d,%s,%d,%d,%d,%d,%.12g,%.12g,%.12g,', ...
            '%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d\n'], ...
            class(Problem),Problem.D,Problem.DU,Problem.DL,generation,mode,i, ...
            genLog(i).Usable,genLog(i).Applied,genLog(i).M,genLog(i).MeanDist, ...
            genLog(i).Rdist,genLog(i).Rdir,genLog(i).Rnum,genLog(i).R, ...
            genLog(i).Lambda,genLog(i).DRawNorm,genLog(i).DeltaRawNorm, ...
            genLog(i).DeltaFinalNorm,genLog(i).SigmaClipped);
    end
    clear cleaner
end
