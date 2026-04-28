function reward = RewardCalculator(Current, NextCurrent, ppo_survival_rate)
% RewardCalculator
% Reward based on population transition: Current -> NextCurrent

    if nargin < 3
        ppo_survival_rate = 0;
    end

    % ===== 1. best fitness improvement =====
    current_best = min(Current.ul_fitness);
    next_best    = min(NextCurrent.ul_fitness);
    r_best = (current_best - next_best) / (abs(current_best) + 1e-8);

    % ===== 2. mean fitness improvement =====
    current_mean = mean(Current.ul_fitness);
    next_mean    = mean(NextCurrent.ul_fitness);
    r_mean = (current_mean - next_mean) / (abs(current_mean) + 1e-8);

    % ===== 3. lower-level feasibility improvement =====
    if isempty(Current.ll_con)
        feas_cur = 1;
    else
        feas_cur = mean(all(Current.ll_con <= 0, 2));
    end

    if isempty(NextCurrent.ll_con)
        feas_next = 1;
    else
        feas_next = mean(all(NextCurrent.ll_con <= 0, 2));
    end
    r_feas = feas_next - feas_cur;

    % ===== 4. diversity term =====
    if size(Current.ulPop,1) == 1
        div_cur = 0;
    else
        div_cur = mean(std(Current.ulPop, 0, 1));
    end

    if size(NextCurrent.ulPop,1) == 1
        div_next = 0;
    else
        div_next = mean(std(NextCurrent.ulPop, 0, 1));
    end

    if abs(next_best) > 1e-3
        r_div = max(0, div_next - 0.8 * div_cur);
    else
        r_div = 0;
    end

    % ===== 5. PPO offspring survival rate =====
    r_survival = 2 * ppo_survival_rate - 1;

    % ===== 6. final reward =====
    reward = 1.0 * r_best + 0.3 * r_mean + 0.2 * r_feas + 0.05 * r_div + 0.15 * r_survival;

    % ===== 7. clipping =====
    reward = max(min(reward, 5), -5);
end
