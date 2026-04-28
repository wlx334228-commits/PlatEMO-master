function Theta = ActionToParam(action, Current, Problem, sigma0)
% ActionToParam
% Map per-dimension PPO actions to upper-level Gaussian parameters.
%
% Action format:
%   action = [delta_mu_1,...,delta_mu_DU, delta_sigma_1,...,delta_sigma_DU]
%
% Output:
%   Theta.mu    : 1 x DU sampling mean
%   Theta.sigma : 1 x DU sampling standard deviation

    ulPop = Current.ulPop;
    fitness = Current.ul_fitness;

    [N, DU] = size(ulPop);

    if length(action) ~= 2 * DU
        error('ActionToParam: action dimension mismatch. Expected %d, got %d.', 2*DU, length(action));
    end

    [~, rank] = sort(fitness, 'ascend');
    elite_num = max(2, ceil(0.2 * N));
    elitePop = ulPop(rank(1:elite_num), :);
    elite_mean = mean(elitePop, 1);

    pop_std = std(ulPop, 0, 1);
    pop_std(pop_std < 1e-6) = 1e-6;

    delta_mu = action(1:DU);
    delta_sigma = action(DU+1:2*DU);

    delta_mu = max(min(delta_mu, 0.5), -0.5);
    delta_sigma = max(min(delta_sigma, 0.5), -2.0);

    mu = elite_mean + delta_mu .* pop_std;
    sigma = sigma0 .* exp(delta_sigma) .* pop_std;

    lower = Problem.lower(1:DU);
    upper = Problem.upper(1:DU);

    mu = min(max(mu, lower), upper);
    sigma = max(sigma, 1e-6);

    Theta.mu = mu;
    Theta.sigma = sigma;
end
%第二版，逐维delta_mu+逐维delta_sigma
% function Theta = ActionToParam(action, Current, Problem, sigma0)
% % ActionToParam
% % Map PPO action to upper-level sampling distribution parameters
% %
% % New action format:
% %   action = [delta_mu_1,...,delta_mu_DU, delta_sigma_1,...,delta_sigma_DU]
% %
% % Output:
% %   Theta.mu
% %   Theta.sigma
% 
%     ulPop = Current.ulPop;
%     fitness = Current.ul_fitness;
% 
%     [N, DU] = size(ulPop);
% 
%     % ===== 1. 按适应度排序 =====
%     [~, rank] = sort(fitness, 'ascend');
% 
%     % ===== 2. 精英集合 =====
%     elite_num = max(2, ceil(0.2 * N));
%     elitePop = ulPop(rank(1:elite_num), :);
% 
%     % ===== 3. 精英中心 =====
%     elite_mean = mean(elitePop, 1);
% 
%     % ===== 4. 种群逐维尺度 =====
%     pop_std = std(ulPop, 0, 1);
%     pop_std(pop_std < 1e-6) = 1e-6;
% 
%     % ===== 5. 解析动作 =====
%     if length(action) ~= 2 * DU
%         error('ActionToParam: action dimension mismatch. Expected %d, got %d.', 2*DU, length(action));
%     end
% 
%     delta_mu = action(1:DU);                 % 1 x DU
%     delta_sigma = action(DU+1 : 2*DU);       % 1 x DU
% 
%     % ===== 6. 截断动作 =====
%     delta_mu = max(min(delta_mu, 0.5), -0.5);
%     delta_sigma = max(min(delta_sigma, 0.5), -2.0);
% 
%     % ===== 7. 构造逐维 mu =====
%     mu = elite_mean + delta_mu .* pop_std;
% 
%     % ===== 8. 构造逐维 sigma =====
%     sigma = sigma0 .* exp(delta_sigma) .* pop_std;
% 
%     % ===== 9. 接近最优时缩小 sigma =====
%     current_best = min(Current.ul_fitness);
% 
%     if Problem.DU <= 2
%         if abs(current_best) < 1e-3
%             sigma = 0.6 * sigma;
%         end
%         if abs(current_best) < 1e-4
%             sigma = 0.4 * sigma;
%         end
%         sigma(sigma < 1e-6) = 1e-6;
%     else
%         if abs(current_best) < 1e-3
%             sigma = 0.9 * sigma;
%         end
%         if abs(current_best) < 1e-4
%             sigma = 0.8 * sigma;
%         end
%         sigma(sigma < 1e-4) = 1e-4;
%     end
% 
%     % ===== 10. 边界修复 =====
%     lower = Problem.lower(1:DU);
%     upper = Problem.upper(1:DU);
% 
%     mu = min(max(mu, lower), upper);
% 
%     Theta.mu = mu;
%     Theta.sigma = sigma;
% end
