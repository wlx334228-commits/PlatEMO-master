function [value, cache] = GetValueFromCritic(state, Critic)

    z1 = state * Critic.W1 + Critic.b1;
    h1 = tanh(z1);

    value = h1 * Critic.W2 + Critic.b2;   % scalar

    cache.state = state;
    cache.z1 = z1;
    cache.h1 = h1;
end