function AppendRunFERecord(problemName,runID,upperFE,totalLowerFE)
% Append final upper/lower FE counters for each run.

    file = fullfile(fileparts(mfilename('fullpath')),'BLRLEA4_run_fe.tsv');
    needHeader = ~exist(file,'file');
    fid = fopen(file,'a');
    if fid < 0
        warning('BLRLEA4:LogOpenFailed','Cannot open FE log file: %s',file);
        return;
    end
    cleaner = onCleanup(@()fclose(fid)); %#ok<NASGU>

    if needHeader
        fprintf(fid,'Problem\tRunID\tUpperFE\tTotalLowerFE\n');
    end
    fprintf(fid,'%s\t%s\t%d\t%d\n',problemName,runID,upperFE,totalLowerFE);
end
