function [best_xl, best_f, best_con, llFunctionEvaluations, ulFunctionEvaluations] = ...
    llSearch(xl0, flag, ulPop, P_llPop, Problem, llDim, llDimMin, llDimMax, ...
             xu, llPopSize, llStoppingCriteria, llFunctionEvaluations, llMaxGens, ulFunctionEvaluations)
% llSearch
% Lower-level search using standard Differential Evolution (DE)
%
% Input:
%   xl0, flag, ulPop, P_llPop : reserved for interface compatibility, not used here
%   Problem                   : PlatEMO problem object
%   llDim                     : lower-level dimension
%   llDimMin, llDimMax        : lower-level bounds
%   xu                        : 1 x DU fixed upper-level solution
%   llPopSize                 : lower-level population size
%   llStoppingCriteria        : stopping threshold based on population contraction
%   llFunctionEvaluations     : current lower-level FE counter
%   llMaxGens                 : maximum lower-level generations
%   ulFunctionEvaluations     : pass-through upper-level FE counter
%
% Output:
%   best_xl
%   best_f
%   best_con
%   llFunctionEvaluations
%   ulFunctionEvaluations

    %#ok<INUSD> % 保留接口一致性，当前普通DE版不使用这些输入
    F  = 0.5;   % mutation factor
    CR = 0.9;   % crossover probability

    % ===== 1. 下层种群随机初始化 =====
    llPop = repmat(llDimMax - llDimMin, llPopSize, 1) .* rand(llPopSize, llDim) + repmat(llDimMin, llPopSize, 1);

    % ===== 2. 初始评价 =====
    [ll_obj, ~, ll_con] = llEvaluate(llPop, Problem, xu);
    llFunctionEvaluations = llFunctionEvaluations + llPopSize;

    ll_fitness = LocalFitness(ll_obj, ll_con);

    % ===== 3. 记录初始方差 =====
    v0 = var(llPop, 0, 1);
    v0(v0 < 1e-12) = 1e-12;

    
    GEN = 0;
    % ===== 记录最优下层目标值历史 =====
    T = 5;
    best_hist = [];
    [~, best_idx0] = min(ll_fitness);
    best_hist(end+1) = ll_obj(best_idx0,:);

    % ===== 4. DE 主循环 =====
    while mean(var(llPop,0,1)) > llStoppingCriteria && GEN <= llMaxGens   %mean(var(llPop,0,1))种群平均每维方差小于1e-5

        Offspring = zeros(llPopSize, llDim);

        for i = 1:llPopSize
            % ---- DE/rand/1 变异 ----
            idx = randperm(llPopSize,3);
            while any(idx == i)
                idx = randperm(llPopSize,3);
            end

            x1 = llPop(idx(1),:);
            x2 = llPop(idx(2),:);
            x3 = llPop(idx(3),:);

            mutant = x1 + F * (x2 - x3);

            % 边界修复
            mutant = min(max(mutant, llDimMin), llDimMax);

            % ---- binomial crossover ----
            trial = llPop(i,:);
            jrand = randi(llDim);

            for j = 1:llDim
                if rand <= CR || j == jrand
                    trial(j) = mutant(j);
                end
            end

            % 边界修复
            trial = min(max(trial, llDimMin), llDimMax);

            Offspring(i,:) = trial;
        end

        % ===== 5. 子代评价 =====
        [off_ll_obj, ~, off_ll_con] = llEvaluate(Offspring, Problem, xu);
        llFunctionEvaluations = llFunctionEvaluations + llPopSize;

        off_ll_fitness = LocalFitness(off_ll_obj, off_ll_con);

        % ===== 6. 一对一选择 =====
        replace = off_ll_fitness < ll_fitness;

        llPop(replace,:) = Offspring(replace,:);
        ll_obj(replace,:) = off_ll_obj(replace,:);
        ll_fitness(replace,:) = off_ll_fitness(replace,:);

        % 同步约束
        if isempty(ll_con) && ~isempty(off_ll_con)
            ll_con = zeros(llPopSize, size(off_ll_con,2));
        elseif ~isempty(ll_con) && isempty(off_ll_con)
            off_ll_con = zeros(llPopSize, size(ll_con,2));
        elseif isempty(ll_con) && isempty(off_ll_con)
            ll_con = [];
            off_ll_con = [];
        end
        if ~isempty(ll_con) && ~isempty(off_ll_con)
            if size(ll_con,2) ~= size(off_ll_con,2)
                error('ll_con and off_ll_con column sizes do not match.');
            end
        end
        if ~isempty(ll_con)
            ll_con(replace,:) = off_ll_con(replace,:);
        end
        % ===== 更新当前代最优下层目标值历史 =====
        [~, best_idx_now] = min(ll_fitness);
        best_hist(end+1) = ll_obj(best_idx_now,:);

        % ===== 停止准则1：下层种群方差小于阈值 =====
        var_now = mean(var(llPop, 0, 1));
        var_stop = var_now < llStoppingCriteria;

        % ===== 停止准则2：最近T代最优下层目标值变化小于1e-5 =====
        obj_stop = false;
        if length(best_hist) >= T
            recent_best = best_hist(end-T+1:end);
            if max(recent_best) - min(recent_best) < 1e-5
                obj_stop = true;
            end
        end

        % ===== 满足任一停止条件则退出 =====
        if var_stop || obj_stop
            break;
        end

        % % ===== 7. 停止准则：种群收缩 =====
        % alpha = sum(var(llPop,0,1) ./ v0);
        % if alpha > 1
        %     alpha = 1;
        % end

        GEN = GEN + 1;
    end

    % ===== 8. 返回当前最优下层解 =====
    [~, best_idx] = min(ll_fitness);

    best_xl = llPop(best_idx,:);
    best_f  = ll_obj(best_idx,:);

    if isempty(ll_con)
        best_con = 0;
    else
        best_con = ll_con(best_idx,:);
    end
end