classdef BLRLEA < ALGORITHM
    methods
        function main(Algorithm,Problem)
            %上下层种群大小
            ulPopSize = floor(Problem.N/2);
            llPopSize = Problem.N-ulPopSize;

            %上下层变量维度
            ulDim = Problem.DU;
            llDim = Problem.DL;

            %总函数评价次数=10000
            maxFE=10000;

            %上下层变量对应维度取值范围
            ulDimMin = Problem.lower(1:Problem.DU);
            ulDimMax = Problem.upper(1:Problem.DU);
            llDimMin = Problem.lower(Problem.DU+1:end);
            llDimMax = Problem.upper(Problem.DU+1:end);

            %下层截止条件
            llStoppingCriteria = 1e-5;

            %下层最大迭代代数
            llMaxGens = Problem.maxFElower;

            %上层种群初始化+下层搜索
            proC=1;disC=20;proM=1;disM=20;

            ulFunctionEvaluations = 0;
            llFunctionEvaluations = 0;

            stateDim = 10;                              %状态维度
            actionDim = 2*ulDim;                              %动作维度: [global_delta_mu, global_delta_sigma]

            hiddenDim = 32;                             %隐藏层
            learnRate = 1e-3;                           %学习率

            %代内训练步数,,,,,
            
            maxstep = 1;

            %PPO生成比例
            ppo_ratio = 0.5;

            %PPO模型生成xu数量与普通交叉变异生成xu数量
            N_ppo = round(ppo_ratio * ulPopSize);
            N_ea  = ulPopSize - N_ppo;

            %基础采样尺度
            sigma0 = 1;
            % ===== 局部精细化搜索参数 =====
            local_refine_gap = 5;        % 每隔5代做一次；想试3代就改成3
            local_refine_num = 0.5*ulPopSize;       % 局部候选个数
            local_refine_scale = 0.2;    % 局部搜索尺度，相对于当前种群std

            stall_count = 0;              % 连续停滞代数
            stall_tol = 1e-7;             % 小于这个改进量，认为停滞
            stall_trigger = 6;            % 连续5代停滞后触发精细化

            Pu = UniformPoint(ulPopSize,ulDim,'Latin');
            ulPop = repmat(ulDimMin,ulPopSize,1) + repmat(ulDimMax-ulDimMin,ulPopSize,1).*Pu;

            %对每一个xu调用下层搜索求得xu对应的下层最优解
            for i=1:ulPopSize
                [llPop(i,:), ll_obj(i,:), ll_con_i, llFunctionEvaluations, ulFunctionEvaluations] = ...
                    llSearch([], 0, [], [], Problem, llDim, llDimMin, llDimMax, ...
                    ulPop(i,:), llPopSize, llStoppingCriteria, ...
                    llFunctionEvaluations, llMaxGens, ulFunctionEvaluations);
                if isempty(ll_con_i)
                    ll_con(i,1) = 0;
                else
                    ll_con(i,1:size(ll_con_i,2)) = ll_con_i;
                end

            end
            % %测试llsearch()是否能找到上层xu对应的下层最优解
            % xu_test = ulPop(1,:);
            % for k = 1:5
            %     [best_xl, best_f, best_con, llFE, ulFE] = llSearch([], 0, [], [], Problem, ...
            %         llDim, llDimMin, llDimMax, xu_test, llPopSize, llStoppingCriteria, ...
            %         0, llMaxGens, 0);
            %     fprintf('run %d: best_f = %.12f\n', k, best_f);
            % end

            %对完整解[xu xl]进行评价
            [ul_obj,~,ul_con] = ulevaluate(ulPop,llPop,Problem);
            ulFunctionEvaluations = ulFunctionEvaluations + ulPopSize;
            if isempty(ul_con)
                [n,~] = size(ul_obj);
                ul_con = zeros(n,1);
            end

            %计算完整解[xu xl]的适应度
            if all(ul_con(:)==0) && all(ll_con(:)==0)
                ul_fitness=ul_obj;
            else
                PopCon   = sum(max(0,ul_con),2)+sum(max(0,ll_con),2);
                Feasible = PopCon <= 0;
                ul_fitness  = Feasible.*ul_obj + ~Feasible.*(PopCon+1e10);
            end

            last_best = min(ul_fitness);
            %初始化种群代数
            GEN = 0;

            %打包个体为Population对象
            %Population = Problem.Evalution([ulPop,llPop]);
            PopDec = [ulPop, llPop];
            PopObj = [ul_obj, ll_obj];
            PopCon = [ul_con, ll_con];
            Population = SOLUTION(PopDec, PopObj, PopCon);


            %初始化Actor与Critic神经网络
            Actor  = InitializeActorNetwork(stateDim, actionDim, hiddenDim, learnRate);
            Critic = InitializeCriticNetwork(stateDim, hiddenDim, learnRate);

            rolloutUpdateGap = 3;
            rolloutGenCount  = 0;
            rolloutBuffer    = InitializePPOBuffer();

            %初始化每一代最优解集合
            History.best_fit = [];
            History.best_ul_obj = [];
            History.best_ll_obj = [];
            History.best_xu = {};
            History.best_xl = {};

            %Iterate
            while abs(min(ul_fitness)) >= 1e-4 && ulFunctionEvaluations < maxFE

                %当前代种群信息用Current对象接收
                PopDec = Population.decs;
                PopObj = Population.objs;

                Current.ulPop = PopDec(:,1:Problem.DU);
                Current.llPop = PopDec(:,Problem.DU+1:end);
                Current.ul_obj = PopObj(:,1);
                Current.ll_obj = PopObj(:,2);
                Current.ul_con = ul_con;
                Current.ll_con = ll_con;
                Current.ul_fitness = ul_fitness;

                % Parent = Current;

                %调用BuildState()获取种群状态信息，信息包括F_best,F_mean,F_std,div_ul,feasible_ll_rate,improve_rate
                % f_best,          % 当前代最优上层目标值
                % f_mean,          % 当前代平均上层目标值
                % f_std,           % 当前代上层目标值标准差
                % div_u,           % 当前代上层种群多样性
                % feas_ll_rate,    % 当前代下层可行率
                % improve_rate     % 相比上一代的最优值改进量
                

                [Actor, Critic, buffer,NextCurrent, llFunctionEvaluations, ulFunctionEvaluations] = ...
                    TrainPPOInGeneration(Actor, Critic, Current, Problem, ...
                    maxstep, ulPopSize, ppo_ratio, sigma0, proC, disC, proM, disM, ...
                    llMaxGens, last_best,llPopSize,llStoppingCriteria,llDimMin,llDimMax, llFunctionEvaluations, ulFunctionEvaluations);

                Current = NextCurrent;
                rolloutBuffer = AppendPPOBuffer(rolloutBuffer, buffer);
                rolloutGenCount = rolloutGenCount + 1;

                if rolloutGenCount >= rolloutUpdateGap
                    rolloutBuffer.dones(end) = 1;
                    [Actor, Critic] = PPOUpdate(Actor, Critic, rolloutBuffer);
                    rolloutBuffer = InitializePPOBuffer();
                    rolloutGenCount = 0;
                end

                % % ===== 训练结束后，回到原始当前代 =====
                % Current = Parent;
                % state = BuildState(Current,last_best);
                % % ===== 再由训练后的PPO输出正式动作 =====
                % [action, log_prob, actionMean, actionStd] = ActorForward(Actor, state);
                % 
                % %统计当代种群精英解分布，与PPO模型输出分布参数相计算，输出最终生成上层xu的分布参数
                % ul_Theta = ActionToParam(action, Current, Problem, sigma0);
                % 
                % %基于分布参数生成最终的上层xu,数量为ppo_ratio*N
                % Offspring_ulpop_ppo = GenerateUpperOffspring(ul_Theta, Problem, N_ppo);
                % 
                % %普通交叉变异生成上层子代xu,数量为N-ppo_ratio*N
                % Offspring_ulpop_ea = GenerateEAOffspring(Current.ulPop, Problem, N_ea, proC, disC, proM, disM);
                % 
                % % 合并offspring_ulpop与offspring_ulpop_ea
                % Offspring_ulpop = [Offspring_ulpop_ppo; Offspring_ulpop_ea];
                % 
                % 
                % % ===== 对每个上层子代做下层优化 =====
                % Offspring_llpop = zeros(size(Offspring_ulpop,1), Problem.DL);
                % off_ll_obj = zeros(size(Offspring_ulpop,1),1);
                % off_ll_con = [];
                % 
                % for i = 1:size(Offspring_ulpop,1)
                %     xu = Offspring_ulpop(i,:);
                % 
                %     [Offspring_llpop(i,:), off_ll_obj(i,:), ll_con_i, llFunctionEvaluations, ulFunctionEvaluations] = ...
                %         llSearch([], 0, [], [], Problem, llDim, llDimMin, llDimMax, ...
                %         xu, llPopSize, llStoppingCriteria, ...
                %         llFunctionEvaluations, llMaxGens, ulFunctionEvaluations);
                % 
                %     if isempty(ll_con_i)
                %         off_ll_con(i,1) = 0;
                %     else
                %         off_ll_con(i,1:size(ll_con_i,2)) = ll_con_i;
                %     end
                % end
                % 
                % % ===== 完整子代评价 =====
                % [off_ul_obj, ~, off_ul_con] = ulevaluate(Offspring_ulpop, Offspring_llpop, Problem);
                % ulFunctionEvaluations = ulFunctionEvaluations + size(Offspring_ulpop,1);
                % 
                % % ===== 子代适应度 =====
                % if isempty(off_ul_con)
                %     [n,~] = size(off_ul_obj);
                %     off_ul_con = zeros(n,1);
                % end
                % 
                % if isempty(off_ll_con)
                %     [n,~] = size(off_ul_obj);
                %     off_ll_con = zeros(n,1);
                % end
                % 
                % if all(off_ul_con(:)==0,'all') && all(off_ll_con(:)==0,'all')
                %     off_ul_fitness = off_ul_obj;
                % else
                %     PopCon   = sum(max(0,off_ul_con),2) + sum(max(0,off_ll_con),2);
                %     Feasible = PopCon <= 0;
                %     off_ul_fitness = Feasible.*off_ul_obj + ~Feasible.*(PopCon + 1e10);
                % end

                % % ===== 父代子代合并 =====
                % Combine_ulPop      = [Current.ulPop; Offspring_ulpop];
                % Combine_llPop      = [Current.llPop; Offspring_llpop];
                % Combine_ul_obj     = [Current.ul_obj; off_ul_obj];
                % Combine_ll_obj     = [Current.ll_obj; off_ll_obj];
                % Combine_ul_con     = [Current.ul_con; off_ul_con];
                % Combine_ll_con     = [Current.ll_con; off_ll_con];
                % Combine_ul_fitness = [Current.ul_fitness; off_ul_fitness];

                % % ===== 环境选择 =====
                % [~, rank] = sort(Combine_ul_fitness, 'ascend');
                % index = rank(1:ulPopSize);
                % 
                % ulPop      = Combine_ulPop(index,:);
                % llPop      = Combine_llPop(index,:);
                % ul_obj     = Combine_ul_obj(index,:);
                % ll_obj     = Combine_ll_obj(index,:);
                % ul_con     = Combine_ul_con(index,:);
                % ll_con     = Combine_ll_con(index,:);
                % ul_fitness = Combine_ul_fitness(index,:);

                % %更新种群，依据父代与子代更新Population
                % %打包个体为Population对象
                % %Population = Problem.Evalution([ulPop,llPop]);
                % PopDec = [ulPop, llPop];
                % PopObj = [ul_obj, ll_obj];
                % PopCon = [ul_con, ll_con];
                % Population = SOLUTION(PopDec, PopObj, PopCon);
                ulPop      = Current.ulPop;
                llPop      = Current.llPop;
                ul_obj     = Current.ul_obj;
                ll_obj     = Current.ll_obj;
                ul_con     = Current.ul_con;
                ll_con     = Current.ll_con;
                ul_fitness = Current.ul_fitness;

                PopDec = [ulPop, llPop];
                PopObj = [ul_obj, ll_obj];
                PopCon = [ul_con, ll_con];
                Population = SOLUTION(PopDec, PopObj, PopCon);

                %依据适应度找到合并后最优解[xu xl],上层与下层最优值[best_F best_f]
                % ===== 环境选择后更新当前种群 =====
                [min_fit, best_idx] = min(ul_fitness);
                %两代最优值改进量
                improve = abs(last_best - min_fit);

                if improve < stall_tol
                    stall_count = stall_count + 1;
                else
                    stall_count = 0;
                end

                best_xu     = ulPop(best_idx,:);
                best_xl     = llPop(best_idx,:);
                best_ul_obj = ul_obj(best_idx,:);
                best_ll_obj = ll_obj(best_idx,:);
                best_ul_con = ul_con(best_idx,:);
                best_ll_con = ll_con(best_idx,:);

                % ===== 接近最优且停滞时，触发局部精细化 =====
                if abs(min_fit) < 1e-5 && stall_count >= stall_trigger
                    fprintf('>>> 执行局部精细化搜索: GEN = %d | best fitness = %.6e | stall_count = %d\n', ...
                        GEN, min_fit, stall_count);


                    % 1. 生成局部上层候选
                    local_std = std(ulPop, 0, 1);
                    local_std(local_std < 1e-6) = 1e-6;
                    if abs(min_fit) < 1e-5
                        local_scale_now = 0.05;
                    elseif abs(min_fit) < 1e-4
                        local_scale_now = 0.1;
                    else
                        local_scale_now = local_refine_scale;
                    end

                    local_sigma = local_scale_now .* local_std;

                    local_ulPop = repmat(best_xu, local_refine_num, 1) + ...
                        randn(local_refine_num, ulDim) .* repmat(local_sigma, local_refine_num, 1);

                    % 边界修复
                    local_ulPop = min(max(local_ulPop, repmat(ulDimMin, local_refine_num, 1)), ...
                        repmat(ulDimMax, local_refine_num, 1));

                    % 2. 对每个局部xu做下层优化
                    local_llPop = zeros(local_refine_num, llDim);
                    local_ll_obj = zeros(local_refine_num, 1);
                    local_ll_con = [];

                    for j = 1:local_refine_num
                        xu_local = local_ulPop(j,:);

                        [local_llPop(j,:), local_ll_obj(j,:), ll_con_j, llFunctionEvaluations, ulFunctionEvaluations] = ...
                            llSearch([], 0, [], [], Problem, llDim, llDimMin, llDimMax, ...
                            xu_local, llPopSize, llStoppingCriteria, ...
                            llFunctionEvaluations, llMaxGens, ulFunctionEvaluations);

                        if isempty(ll_con_j)
                            local_ll_con(j,1) = 0;
                        else
                            local_ll_con(j,1:size(ll_con_j,2)) = ll_con_j;
                        end
                    end

                    % 3. 完整评价局部候选
                    [local_ul_obj, ~, local_ul_con] = ulevaluate(local_ulPop, local_llPop, Problem);
                    ulFunctionEvaluations = ulFunctionEvaluations + local_refine_num;

                    if isempty(local_ul_con)
                        local_ul_con = zeros(local_refine_num,1);
                    end
                    if isempty(local_ll_con)
                        local_ll_con = zeros(local_refine_num,1);
                    end

                    % 4. 计算局部候选适应度
                    if all(local_ul_con(:)==0) && all(local_ll_con(:)==0)
                        local_ul_fitness = local_ul_obj;
                    else
                        LocalCon = sum(max(0,local_ul_con),2) + sum(max(0,local_ll_con),2);
                        LocalFeasible = LocalCon <= 0;
                        local_ul_fitness = LocalFeasible .* local_ul_obj + ~LocalFeasible .* (LocalCon + 1e10);
                    end

                    % 5. 将局部精细化个体与当前种群整体合并
                    Combine2_ulPop      = [ulPop; local_ulPop];
                    Combine2_llPop      = [llPop; local_llPop];
                    Combine2_ul_obj     = [ul_obj; local_ul_obj];
                    Combine2_ll_obj     = [ll_obj; local_ll_obj];
                    Combine2_ul_con     = [ul_con; local_ul_con];
                    Combine2_ll_con     = [ll_con; local_ll_con];
                    Combine2_ul_fitness = [ul_fitness; local_ul_fitness];

                    % 6. 再做一次环境选择
                    [~, rank2] = sort(Combine2_ul_fitness, 'ascend');
                    index2 = rank2(1:ulPopSize);

                    ulPop      = Combine2_ulPop(index2,:);
                    llPop      = Combine2_llPop(index2,:);
                    ul_obj     = Combine2_ul_obj(index2,:);
                    ll_obj     = Combine2_ll_obj(index2,:);
                    ul_con     = Combine2_ul_con(index2,:);
                    ll_con     = Combine2_ll_con(index2,:);
                    ul_fitness = Combine2_ul_fitness(index2,:);

                    % 7. 更新 Population
                    PopDec = [ulPop, llPop];
                    PopObj = [ul_obj, ll_obj];
                    PopCon = [ul_con, ll_con];
                    Population = SOLUTION(PopDec, PopObj, PopCon);

                    % 8. 重新确定当前代最优个体
                    [min_fit, best_idx] = min(ul_fitness);

                    best_xu     = ulPop(best_idx,:);
                    best_xl     = llPop(best_idx,:);
                    best_ul_obj = ul_obj(best_idx,:);
                    best_ll_obj = ll_obj(best_idx,:);
                    best_ul_con = ul_con(best_idx,:);
                    best_ll_con = ll_con(best_idx,:);

                    fprintf('>>> 精细化结束: GEN = %d | refined best fitness = %.6e\n', GEN, min_fit);

                    % 9. 只要执行过精细化，就把停滞计数清零
                    stall_count = 0;
                end

               

                % ===== 保存历史 =====
                History.best_fit(end+1)    = min_fit;
                History.best_ul_obj(end+1) = best_ul_obj;
                History.best_ll_obj(end+1) = best_ll_obj;
                History.best_xu{end+1}     = best_xu;
                History.best_xl{end+1}     = best_xl;

                % ===== 输出每代信息 =====
                fprintf('GEN = %d | best fitness = %.6e | best ul_obj = %.6e | best ll_obj = %.6e\n', ...
                    GEN, min_fit, best_ul_obj, best_ll_obj);

                fprintf('best xu = [');
                fprintf(' %.4f', best_xu);
                fprintf(' ]\n');

                fprintf('best xl = [');
                fprintf(' %.4f', best_xl);
                fprintf(' ]\n');

                fprintf(' 上层函数评估次数为：%.4f', ulFunctionEvaluations);
                fprintf(' \n');
                fprintf(' 下层函数评估次数为：%.4f', llFunctionEvaluations);
                fprintf(' \n');

                % ===== 更新上一代最优适应度 =====
                last_best = min_fit;

                % ===== 代数加1 =====
                GEN = GEN + 1;

            end

            if rolloutGenCount > 0 && ~isempty(rolloutBuffer.states)
                rolloutBuffer.dones(end) = 1;
                [Actor, Critic] = PPOUpdate(Actor, Critic, rolloutBuffer);
            end
        end
    end
end

