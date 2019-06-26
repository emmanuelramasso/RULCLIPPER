function FBeta = fbetamesure(precis_val, rappel_val, betavaln) 
% Compute the Fbeta measure, called also F1 score
% precis_val = precision
% rappel_val = recall
% betavaln = in [0,1]
% https://en.wikipedia.org/wiki/F1_score
% 
% Author: Emmanuel Ramasso, 2014
%

FBeta = ((1+(betavaln^2)).*precis_val.*rappel_val) ./ ((betavaln^2) .* precis_val + rappel_val);
