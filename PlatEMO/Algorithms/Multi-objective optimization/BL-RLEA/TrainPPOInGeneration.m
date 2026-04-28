function [Actor, Critic, buffer,NextCurrent, llFunctionEvaluations, ulFunctionEvaluations] = ...
    TrainPPOInGeneration(Actor, Critic, Current, Problem, ...
    maxstep, N, ppo_ratio, sigma0, proC, disC, proM, disM, ...
    llBudget, last_best,llPopSize,llStoppingCriteria, llDimMin,llDimMax,llFunctionEvaluations, ulFunctionEvaluations)

buffer.states      = [];
buffer.actions     = [];
buffer.rewards     = [];
buffer.next_states = [];
buffer.log_probs   = [];
buffer.values      = [];
buffer.action_means = [];
buffer.action_stds  = [];
buffer.dones        = [];

state = BuildState(Current, last_best);

for t = 1:maxstep

    % ===== 1. PPO生成部分 =====
    N_ppo = round(ppo_ratio * N);
    N_ea  = (N - N_ppo);

    [action, log_prob, action_mean, action_std] = ActorForward(Actor, state);
    Theta = ActionToParam(action, Current, Problem, sigma0);
    temp_ulPop_ppo = GenerateUpperOffspring(Theta, Problem, N_ppo);

    % ===== 2. 普通交叉变异部分 =====
    temp_ulPop_ea = GenerateEAOffspring(Current.ulPop, Problem, N_ea, proC, disC, proM, disM);

    % ===== 3. 合并临时上层候选 =====
    temp_ulPop = [temp_ulPop_ppo; temp_ulPop_ea];

    % ===== 4. 下层优化 =====
    % temp_llPop = zeros(size(temp_ulPop,1), Problem.DL);
    % for i = 1:size(temp_ulPop,1)
    %     temp_llPop(i,:) = llSearch(temp_ulPop(i,:), Problem, llBudget);
    % end
    temp_llPop = zeros(size(temp_ulPop,1), Problem.DL);
    temp_ll_obj = zeros(size(temp_ulPop,1), 1);
    temp_ll_con = [];

    for i = 1:size(temp_ulPop,1)
        xu = temp_ulPop(i,:);

        [temp_llPop(i,:), temp_ll_obj(i,:), ll_con_i, llFunctionEvaluations, ulFunctionEvaluations] = ...
            llSearch([], 0, [], [], Problem, Problem.DL, ...
            llDimMin,llDimMax,xu, llPopSize, llStoppingCriteria, llFunctionEvaluations, llBudget, ulFunctionEvaluations);

        if isempty(ll_con_i)
            temp_ll_con(i,1) = 0;
        else
            temp_ll_con(i,1:size(ll_con_i,2)) = ll_con_i;
        end
    end

    % ===== 5. 完整解评价 =====
    % TempPopulation = Problem.Evaluation([temp_ulPop, temp_llPop]);
    [temp_ul_obj, ~, temp_ul_con] = ulevaluate(temp_ulPop, temp_llPop, Problem);
    ulFunctionEvaluations = ulFunctionEvaluations + (N_ppo+N_ea);

    % %将完整解保存为TempPopulation，SOLUTION对象
    % TempPopDec = [temp_ulPop, temp_llPop];
    % TempPopObj = [temp_ul_obj, temp_ll_obj];
    % TempPopCon = [temp_ul_con, temp_ll_con];
    % TempPopulation = SOLUTION(TempPopDec, TempPopObj, TempPopCon);
    % TempObj = TempPopulation.objs;
    % temp_ul_obj = TempObj(:,1);
    % temp_ll_obj = TempObj(:,2);

    % [~,~,temp_ul_con] = uLevaluate(temp_ulPop, temp_llPop, Problem);
    % [~,~,temp_ll_con] = llEvaluate(temp_llPop, Problem, temp_ulPop);
    
    
 

    if isempty(temp_ul_con)
        temp_ul_con = zeros(size(temp_ul_obj,1),1);
    end
    if isempty(temp_ll_con)
        temp_ll_con = zeros(size(temp_ul_obj,1),1);
    end

    PopCon = sum(max(0,temp_ul_con),2) + sum(max(0,temp_ll_con),2);
    Feasible = PopCon <= 0;
    temp_ul_fitness = Feasible.*temp_ul_obj + ~Feasible.*(PopCon + 1e10);

    % Reward learning only uses the PPO-generated offspring branch.
    PPOCurrent.ulPop      = temp_ulPop_ppo;
    PPOCurrent.llPop      = temp_llPop(1:N_ppo,:);
    PPOCurrent.ul_obj     = temp_ul_obj(1:N_ppo,:);
    PPOCurrent.ll_obj     = temp_ll_obj(1:N_ppo,:);
    PPOCurrent.ul_con     = temp_ul_con(1:N_ppo,:);
    PPOCurrent.ll_con     = temp_ll_con(1:N_ppo,:);
    PPOCurrent.ul_fitness = temp_ul_fitness(1:N_ppo,:);

    % ===== 6. 父代 + 子代环境选择 =====
    Combine_ulPop      = [Current.ulPop; temp_ulPop];
    Combine_llPop      = [Current.llPop; temp_llPop];
    Combine_ul_obj     = [Current.ul_obj; temp_ul_obj];
    Combine_ll_obj     = [Current.ll_obj; temp_ll_obj];
    Combine_ul_con     = [Current.ul_con; temp_ul_con];
    Combine_ll_con     = [Current.ll_con; temp_ll_con];
    Combine_ul_fitness = [Current.ul_fitness; temp_ul_fitness];

    [~, rank] = sort(Combine_ul_fitness, 'ascend');
    index = rank(1:N);

    current_num = size(Current.ulPop,1);
    ppo_survivor_num = sum(index > current_num & index <= current_num + N_ppo);
    ppo_survival_rate = ppo_survivor_num / max(N_ppo,1);

    NewCurrent.ulPop      = Combine_ulPop(index,:);
    NewCurrent.llPop      = Combine_llPop(index,:);
    NewCurrent.ul_obj     = Combine_ul_obj(index,:);
    NewCurrent.ll_obj     = Combine_ll_obj(index,:);
    NewCurrent.ul_con     = Combine_ul_con(index,:);
    NewCurrent.ll_con     = Combine_ll_con(index,:);
    NewCurrent.ul_fitness = Combine_ul_fitness(index,:);

    % ===== 7. reward =====
    reward = RewardCalculator(Current, PPOCurrent, ppo_survival_rate);

    % ===== 8. next state =====
    next_state = BuildState(NewCurrent, min(Current.ul_fitness));

    % ===== 9. 保存轨迹 =====
    % value = GetValueFromCritic(state, Critic);
    value = CriticForward(Critic, state);

    buffer.states      = [buffer.states; state];
    buffer.actions     = [buffer.actions; action];
    buffer.rewards     = [buffer.rewards; reward];
    buffer.next_states = [buffer.next_states; next_state];
    buffer.log_probs   = [buffer.log_probs; log_prob];
    buffer.values      = [buffer.values; value];
    buffer.action_means = [buffer.action_means; action_mean];
    buffer.action_stds  = [buffer.action_stds; action_std];
    buffer.dones        = [buffer.dones; 0];
    % ===== 9. 推进 =====
    state = next_state;
    Current = NewCurrent;
end
NextCurrent = Current;

end
