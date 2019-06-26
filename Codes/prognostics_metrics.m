function [S FPR FNR MAPE MAE MSE Acc] = prognostics_metrics(trueRUL, pred, earlyPenalty, latePenalty)
% This file generate some prognostics metrics, initially developped for
% RULCLIPPER algorithm proposed in [1]. Many metrics have been provided, see 
% for instance the works of A. Saxena et al. [2]. 
%
% Inputs: trueRUL is the true RUL (remaining useful life)
% pred is the estimated RUL, earlyPenalty and latePenalty are two coeffients used in the
% timeliness measure. Both are column vectors. 
% In the PHM challenge and turbofan datasets, see [1] and references therein:
% earlyPenalty = 13 and returns exp(-(pred-trueRUL)/13)-1, 
% and latePenalty = 10 and returns exp(+(pred-trueRUL)/10)-1. Late
% predictions are MORE penalized. Illustration:
% figure,plot(-50:1:50, [exp((50:-1:0)/13)-1 exp((1:1:50)/10)-1])
% or
% x=-50:1:50; figure,plot(x,prognostics_metrics(zeros(length(x),1),x(:),13,10))
% title('Timeliness wrt early (<0) or late (>0) predictions')
%
% Outputs:
% S = timeliness measure 
% FPR = false positive rate
% FNR = false negative rate
% MAPE = mean average percentage error
% MAE = mean average error
% MSE = mean squared error
% Acc = accuracy
%
% In case of problems, please contact me.
%
% Author: E. Ramasso (emmanuel.ramasso@femto-st.fr), Jan. 2015
%
% References
%
% [1] E. Ramasso, Investigating computational geometry for failure prognostics,
% International Journal on Prognostics and Health Management, 5(5):1-18, 2014.
%
% [2] Saxena, A. ; Celaya, J. ; Balaban, E. ; Goebel, K. ; Saha, B. ; Saha, S. ; Schwabacher, M.
% Metrics for evaluating performance of prognostic techniques, 
% Prognostics and Health Management, 2008. PHM 2008. International Conference on, pp 1-17.
% 


if size(trueRUL,1)~=size(pred,1), error('PB DIMENSION'), end

% definition dans WANG IEEE 
% opposee de Saxena

d = pred - trueRUL;
S = zeros(size(d));

% LATE prediction, pred > trueRUL, pred - trueRUL > 0
f = find(d >= 0);
FNR = length(f) / length(trueRUL); % false negative rate
S(f) = exp(d(f)/latePenalty)-1;

% EARLY
f = find(d < 0);
FPR = length(f) / length(trueRUL);% false positive rate
S(f) = exp(-d(f)/earlyPenalty)-1;

MAPE = mean(abs((trueRUL-pred)./trueRUL))*100;

MAE = mean(abs(d));

MSE = mean(d.^2);

Acc = 100*length(find(d <= latePenalty & d >= -earlyPenalty))/length(trueRUL);
