function buffer = InitializePPOBuffer()
% Initialize rollout buffer for BLRLEA4 PPO updates.

    buffer.states = [];
    buffer.actions = [];
    buffer.rewards = [];
    buffer.next_states = [];
    buffer.log_probs = [];
    buffer.values = [];
    buffer.dones = [];
    buffer.action_means = [];
    buffer.action_stds = [];
    buffer.elite_mu_targets = [];
    buffer.elite_sigma_targets = [];
end
