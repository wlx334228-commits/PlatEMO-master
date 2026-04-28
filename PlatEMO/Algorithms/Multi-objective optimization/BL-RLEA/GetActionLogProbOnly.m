function [action, log_prob, action_mean, action_std, cache] = GetActionLogProbOnly(state, action, Actor)
% GetActionLogProbOnly
% Recompute log_prob of an existing action under current Actor
%
% Input:
%   state   : 1 x state_dim
%   action  : 1 x action_dim
%   Actor   : actor struct
%
% Output:
%   action       : same as input
%   log_prob     : scalar
%   action_mean  : 1 x action_dim
%   action_std   : 1 x action_dim
%   cache        : intermediate variables for gradient update

    z1 = state * Actor.W1 + Actor.b1;      % 1 x hidden_dim
    h1 = tanh(z1);                         % 1 x hidden_dim

    action_mean = h1 * Actor.W2 + Actor.b2;   % 1 x action_dim
    action_std  = exp(Actor.log_std);         % 1 x action_dim

    var = action_std.^2;
    log_prob = -0.5 * sum(((action - action_mean).^2) ./ var + log(2*pi*var));

    cache.state = state;
    cache.z1 = z1;
    cache.h1 = h1;
    cache.action_mean = action_mean;
    cache.action_std = action_std;
end