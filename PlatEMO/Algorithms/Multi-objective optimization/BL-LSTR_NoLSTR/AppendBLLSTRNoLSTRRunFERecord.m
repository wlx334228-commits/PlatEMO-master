function AppendBLLSTRNoLSTRRunFERecord(Problem,upperFE,totalFElower,generation,upperReached)
% Append one run-level FE record for BLLSTR_NoLSTR.

    folder = fileparts(mfilename('fullpath'));
    file = fullfile(folder,'BLLSTR_NoLSTR_run_fe.tsv');
    writeHeader = ~exist(file,'file');

    fid = fopen(file,'a');
    if fid < 0
        return;
    end

    cleaner = onCleanup(@()fclose(fid));
    if writeHeader
        fprintf(fid,'Problem\tUpperFE\tTotalLowerFE\tGeneration\tUpperReached\n');
    end

    fprintf(fid,'%s\t%d\t%d\t%d\t%d\n', ...
        class(Problem),upperFE,totalFElower,generation,double(upperReached));
end
