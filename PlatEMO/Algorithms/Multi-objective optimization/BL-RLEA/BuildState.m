function state = BuildState(Population, last_best)
% Build RL state from current generation
%
% Input:
%   Population.ulPop      : N x DU upper-level population
%   Population.ul_fitness : N x 1 upper-level fitness values
%   Population.ul_con     : N x C_u upper-level constraints
%   Population.ll_con     : N x C_l lower-level constraints
%   last_best             : best upper-level fitness of previous generation
%
% Output:
%   state                 : 1 x 10 state vector
%
% State definition:
%   [best_fit, mean_fit, std_fit, div_u, feas_ul_rate, feas_ll_rate, ...
%    improve_rate, elite_gap, elite_div, mean_cv]

    if isfield(Population,'ul_fitness') && ~isempty(Population.ul_fitness)
        ul_fit = Population.ul_fitness(:);
    else
        ul_fit = Population.ul_obj(:);
    end

    ulPop = Population.ulPop;

    if isfield(Population,'ul_con') && ~isempty(Population.ul_con)
        ul_con = Population.ul_con;
    else
        ul_con = zeros(size(ulPop,1),1);
    end

    if isfield(Population,'ll_con') && ~isempty(Population.ll_con)
        ll_con = Population.ll_con;
    else
        ll_con = zeros(size(ulPop,1),1);
    end

    best_fit = min(ul_fit);
    mean_fit = mean(ul_fit);

    if isscalar(ul_fit)
        std_fit = 0;
    else
        std_fit = std(ul_fit);
    end

    if size(ulPop,1) == 1
        div_u = 0;
    else
        div_u = mean(std(ulPop,0,1));
    end

    feas_ul_rate = mean(all(ul_con <= 0, 2));
    feas_ll_rate = mean(all(ll_con <= 0, 2));

    if isempty(last_best)
        improve_rate = 0;
    else
        improve_rate = last_best - best_fit;
    end

    [~, rank] = sort(ul_fit, 'ascend');
    elite_num = max(2, ceil(0.2 * numel(rank)));
    elite_idx = rank(1:elite_num);

    elite_fit_mean = mean(ul_fit(elite_idx));
    elite_gap = mean_fit - elite_fit_mean;

    elitePop = ulPop(elite_idx,:);
    if size(elitePop,1) == 1
        elite_div = 0;
    else
        elite_div = mean(std(elitePop,0,1));
    end

    mean_cv = mean(sum(max(0,ul_con),2) + sum(max(0,ll_con),2));

    state = [signedLog(best_fit), ...
             signedLog(mean_fit), ...
             log1p(max(0, std_fit)), ...
             log1p(max(0, div_u)), ...
             2 * feas_ul_rate - 1, ...
             2 * feas_ll_rate - 1, ...
             signedLog(improve_rate), ...
             signedLog(elite_gap), ...
             log1p(max(0, elite_div)), ...
             signedLog(mean_cv)];

    state = max(min(state, 5), -5);
end

function y = signedLog(x)
    y = sign(x) .* log1p(abs(x));
end
