function results = run_BLCMAES_LSTR_SMD_experiment(varargin)
%RUN_BLCMAES_LSTR_SMD_EXPERIMENT Run BLCMAES-LSTR and NoLSTR on SMD1-SMD12.
%
% The saved CSV contains one row for each algorithm/problem/run:
%   Algorithm, Problem, D, ProblemSize, Run, UpperError, LowerError, UpperFE, LowerFE, TotalFE
%
% UpperError is UAcc = abs(best upper objective - true upper optimum).
% LowerError is LAcc = abs(best lower objective - true lower optimum).
% LowerFE is the total lower-level evaluation count printed by the algorithm.
% TotalFE is UpperFE + LowerFE.
% Only the final metric line of each run is saved. Per-generation results are
% captured temporarily for parsing but are not saved unless saveLogs is true.
%
% Examples:
%   results = run_BLCMAES_LSTR_SMD_experiment;
%   results = run_BLCMAES_LSTR_SMD_experiment('runs',30,'D',[5 10 20]);
%   results = run_BLCMAES_LSTR_SMD_experiment('outputDir','D:\results');
%
% D = 5, 10, 20 correspond to SMD dimensions (2+3), (5+5), (10+10).

    opts = parseOptions(varargin{:});

    repoRoot    = fileparts(mfilename('fullpath'));
    platemoRoot = fullfile(repoRoot,'PlatEMO');
    startDir    = pwd;
    cleanupObj  = onCleanup(@() cd(startDir)); %#ok<NASGU>

    cd(repoRoot);
    addpath(genpath(platemoRoot));

    if isempty(opts.outputDir)
        opts.outputDir = fullfile(repoRoot,'experiment_results');
    end
    if ~exist(opts.outputDir,'dir')
        mkdir(opts.outputDir);
    end

    timestamp = datestr(now,'yyyymmdd_HHMMSS');
    csvFile = fullfile(opts.outputDir, ...
        sprintf('BLCMAES_LSTR_SMD_%s_runs%d_%s.csv',formatDTag(opts.D),opts.runs,timestamp));
    logDir = fullfile(opts.outputDir,'logs');

    algorithms = {@BLCMAESLSTR,@BLCMAESLSTR_NoLSTR};
    algorithmNames = {'BLCMAES-LSTR','BLCMAES-LSTR_NoLSTR'};
    problems = {@SMD1,@SMD2,@SMD3,@SMD4,@SMD5,@SMD6, ...
                @SMD7,@SMD8,@SMD9,@SMD10,@SMD11,@SMD12};
    problemNames = {'SMD1','SMD2','SMD3','SMD4','SMD5','SMD6', ...
                    'SMD7','SMD8','SMD9','SMD10','SMD11','SMD12'};

    totalRows = numel(opts.D)*numel(algorithms)*numel(problems)*opts.runs;
    Algorithm  = cell(totalRows,1);
    Problem    = cell(totalRows,1);
    D          = zeros(totalRows,1);
    ProblemSize = cell(totalRows,1);
    Run        = zeros(totalRows,1);
    UpperError = nan(totalRows,1);
    LowerError = nan(totalRows,1);
    UpperFE    = nan(totalRows,1);
    LowerFE    = nan(totalRows,1);
    TotalFE    = nan(totalRows,1);

    row = 0;
    silentOutput = @SilentPlatEMOOutput;
    fprintf('Saving per-run results to:\n%s\n\n',csvFile);

    for dIndex = 1 : numel(opts.D)
        dValue = opts.D(dIndex);
        sizeLabel = dimensionLabel(dValue);
        for a = 1 : numel(algorithms)
            for p = 1 : numel(problems)
                for runNo = 1 : opts.runs
                    row = row + 1;
                    Algorithm{row} = algorithmNames{a};
                    Problem{row}   = problemNames{p};
                    D(row)         = dValue;
                    ProblemSize{row} = sizeLabel;
                    Run(row)       = runNo;

                    fprintf('[%4d/%4d] D=%d %s | %s on %s, run %d/%d...\n', ...
                        row,totalRows,dValue,sizeLabel,algorithmNames{a},problemNames{p},runNo,opts.runs);

                    runLog = '';
                    try
                        runLog = evalc('platemo(''algorithm'',algorithms{a},''problem'',problems{p},''D'',dValue,''save'',0,''run'',runNo,''outputFcn'',silentOutput);');
                        metrics = parseFinalMetrics(runLog);

                        UpperError(row) = metrics.UpperError;
                        LowerError(row) = metrics.LowerError;
                        UpperFE(row)    = metrics.UpperFE;
                        LowerFE(row)    = metrics.LowerFE;
                        TotalFE(row)    = metrics.TotalFE;

                        fprintf('          UAcc=%.6e, LAcc=%.6e, UpperFE=%d, LowerFE=%d, TotalFE=%d\n', ...
                            UpperError(row),LowerError(row),UpperFE(row),LowerFE(row),TotalFE(row));

                        if opts.saveLogs
                            writeRunLog(logDir,algorithmNames{a},problemNames{p},dValue,runNo,runLog,[]);
                        end
                    catch err
                        warning('Run failed: D=%d %s on %s run %d. %s', ...
                            dValue,algorithmNames{a},problemNames{p},runNo,err.message);
                        if opts.saveLogs
                            writeRunLog(logDir,algorithmNames{a},problemNames{p},dValue,runNo,runLog,err);
                        end
                    end

                    results = makeResultTable(Algorithm,Problem,D,ProblemSize,Run, ...
                        UpperError,LowerError,UpperFE,LowerFE,TotalFE,row);
                    writetable(results,csvFile);
                end
            end
        end
    end

    results = makeResultTable(Algorithm,Problem,D,ProblemSize,Run, ...
        UpperError,LowerError,UpperFE,LowerFE,TotalFE,row);
    fprintf('\nDone. Final CSV saved to:\n%s\n',csvFile);
end

function opts = parseOptions(varargin)
    parser = inputParser;
    parser.addParameter('runs',30,@isPositiveScalar);
    parser.addParameter('D',[5 10 20],@isPositiveVector);
    parser.addParameter('outputDir','',@isCharOrString);
    parser.addParameter('saveLogs',false,@isLogicalScalar);
    parser.parse(varargin{:});

    opts = parser.Results;
    opts.runs = round(opts.runs);
    opts.D = round(opts.D);
    if isstring(opts.outputDir)
        opts.outputDir = char(opts.outputDir);
    end
end

function tf = isPositiveScalar(x)
    tf = isnumeric(x) && isscalar(x) && isfinite(x) && x > 0;
end

function tf = isPositiveVector(x)
    tf = isnumeric(x) && isvector(x) && all(isfinite(x)) && all(x > 0);
end

function tf = isCharOrString(x)
    tf = ischar(x) || (isstring(x) && isscalar(x));
end

function tf = isLogicalScalar(x)
    tf = (islogical(x) || isnumeric(x)) && isscalar(x);
end

function results = makeResultTable(Algorithm,Problem,D,ProblemSize,Run, ...
    UpperError,LowerError,UpperFE,LowerFE,TotalFE,row)

    results = table(Algorithm(1:row),Problem(1:row),D(1:row),ProblemSize(1:row),Run(1:row), ...
        UpperError(1:row),LowerError(1:row),UpperFE(1:row),LowerFE(1:row),TotalFE(1:row), ...
        'VariableNames',{'Algorithm','Problem','D','ProblemSize','Run', ...
        'UpperError','LowerError','UpperFE','LowerFE','TotalFE'});
end

function metrics = parseFinalMetrics(runLog)
    lines = regexp(runLog,'\r\n|\n|\r','split');
    metricMask = contains(lines,'UpperFE=') & ...
                 contains(lines,'TotalLowerFE=') & ...
                 contains(lines,'UAcc=') & ...
                 contains(lines,'LAcc=');
    metricLines = lines(metricMask);
    if isempty(metricLines)
        error('No metric line containing UpperFE, TotalLowerFE, UAcc, and LAcc was found.');
    end

    finalLine = metricLines{end};
    metrics.UpperError = parseNumberField(finalLine,'UAcc');
    metrics.LowerError = parseNumberField(finalLine,'LAcc');
    metrics.UpperFE    = round(parseNumberField(finalLine,'UpperFE'));
    metrics.LowerFE    = round(parseNumberField(finalLine,'TotalLowerFE'));
    metrics.TotalFE    = metrics.UpperFE + metrics.LowerFE;
end

function value = parseNumberField(line,fieldName)
    expr = [fieldName,'\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'];
    token = regexp(line,expr,'tokens','once');
    if isempty(token)
        error('The field "%s" was not found in the final metric line: %s',fieldName,line);
    end
    value = str2double(token{1});
end

function writeRunLog(logDir,algorithmName,problemName,dValue,runNo,runLog,err)
    if ~exist(logDir,'dir')
        mkdir(logDir);
    end
    fileName = sprintf('%s_%s_D%d_run%03d.log', ...
        sanitizeFileName(algorithmName),sanitizeFileName(problemName),dValue,runNo);
    filePath = fullfile(logDir,fileName);
    fid = fopen(filePath,'w');
    if fid < 0
        warning('Failed to write log file: %s',filePath);
        return;
    end
    cleaner = onCleanup(@() fclose(fid));
    if ~isempty(err)
        fprintf(fid,'ERROR:\n%s\n\n',getReport(err,'extended','hyperlinks','off'));
    end
    fprintf(fid,'%s',runLog);
    clear cleaner;
end

function name = sanitizeFileName(name)
    name = regexprep(name,'[^A-Za-z0-9_-]','_');
end

function tag = formatDTag(dValues)
    parts = arrayfun(@(d)sprintf('D%d',d),dValues,'UniformOutput',false);
    tag = strjoin(parts,'_');
end

function label = dimensionLabel(dValue)
    du = floor(dValue/2);
    dl = dValue - du;
    label = sprintf('(%d+%d)',du,dl);
end

function SilentPlatEMOOutput(~,~)
end
