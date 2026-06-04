function Actor = SupervisedActorUpdate(Actor,state,eliteDist,epochs)
% Imitate the current elite distribution before PPO sampling.

    if epochs <= 0
        return;
    end
    if ~isfield(Actor,'step') || isempty(Actor.step)
        Actor.step = 0;
    end
    if isempty(Actor.avgGrad)
        Actor.avgGrad = [];
    end
    if isempty(Actor.avgSqGrad)
        Actor.avgSqGrad = [];
    end

    state = single(state);
    targetMu = single(eliteDist.muNorm);
    targetSigma = single(eliteDist.sigmaNorm);

    for i = 1 : epochs
        [~,gradients] = dlfeval(@ImitationLoss,Actor.net,state,targetMu,targetSigma, ...
            Actor.DU,single(Actor.sigmaMinNorm),single(Actor.sigmaMaxNorm));

        Actor.step = Actor.step + 1;
        [Actor.net,Actor.avgGrad,Actor.avgSqGrad] = adamupdate( ...
            Actor.net,gradients,Actor.avgGrad,Actor.avgSqGrad,Actor.step,Actor.learnRate);
    end
end

function [loss,gradients] = ImitationLoss(net,state,targetMu,targetSigma,DU,sigmaMinNorm,sigmaMaxNorm)
    dlState = dlarray(state(:),"CB");
    [dlActionMean,~] = forward(net,dlState,Outputs={'actionMean','actionLogStd'});

    rawMu = dlActionMean(1:DU,:);
    rawSigma = dlActionMean(DU+1:2*DU,:);

    dlMuNorm = 1 ./ (1 + exp(-rawMu));
    dlSigmaMin = dlarray(sigmaMinNorm(:),"CB");
    dlSigmaMax = dlarray(sigmaMaxNorm(:),"CB");
    dlSigmaNorm = dlSigmaMin + ...
        (1 ./ (1 + exp(-rawSigma))) .* (dlSigmaMax - dlSigmaMin);

    dlTargetMu = dlarray(targetMu(:),"CB");
    dlTargetSigma = dlarray(targetSigma(:),"CB");

    muLoss = mean((dlMuNorm - dlTargetMu).^2,'all');
    sigmaLoss = mean((log(dlSigmaNorm + 1e-12) - ...
        log(dlTargetSigma + 1e-12)).^2,'all');
    loss = muLoss + 0.5 * sigmaLoss;
    gradients = dlgradient(loss,net.Learnables);
end
