function [logProb, entropy, actionMean, actionStd] = EvaluateAction(Actor, state, action)
% EvaluateAction
% Evaluate an existing action under the current Actor policy
%
% Input:
%   Actor      : actor struct from InitializeActorNetwork
%   state      : 1 x stateDim numeric vector
%   action     : 1 x actionDim numeric vector
%
% Output:
%   logProb    : scalar log probability of the given action
%   entropy    : scalar entropy of the Gaussian policy
%   actionMean : 1 x actionDim mean of Gaussian policy
%   actionStd  : 1 x actionDim std of Gaussian policy

    % ===== 1. state -> dlarray =====
    dlState = dlarray(single(state(:)), "CB");

    % ===== 2. forward actor network to get action mean and state-dependent std =====
    [dlActionMean, dlActionLogStd] = forward(Actor.net, dlState, Outputs={'actionMean','actionLogStd'});
    dlActionLogStd = max(min(dlActionLogStd, 1), -5);
    actionMean = extractdata(dlActionMean)';      % 1 x actionDim
    actionStd = exp(extractdata(dlActionLogStd))';   % 1 x actionDim
    actionStd = max(actionStd, 1e-6);

    % ===== 4. compute log probability of given action =====
    var = actionStd .^ 2;
    logProb = -0.5 * sum(((action - actionMean).^2) ./ var + log(2*pi*var));

    % ===== 5. compute entropy of diagonal Gaussian =====
    % H = 0.5 * sum(log(2*pi*e*sigma^2))
    entropy = 0.5 * sum(log(2*pi*exp(1)*var));
end
