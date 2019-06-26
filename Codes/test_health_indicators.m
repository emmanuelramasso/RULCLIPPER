% DOWNLOAD FIRST THE DATASET ON NASA PCOE
% put it in a folder, the function below will ask for the folder path

% estimate HI on dataset 2 using column sensors [7 8 9 10 12 14 16 17 20 25 26]
% with 6 regimes and using the global computation. Local computation could be better
% for dataset #2 or #4, but global can be interested to tackle. It is used in a method described in [1]:
[A B C] = health_indicators_estimation_cmapss(2, [7 8 9 10 12 14 16 17 20 25 26], 6, true);

%this is the response of model 3 for the testing data 1 (of length 258) in dataset 2 obtained above
figure,plot(A{1}(:,3)),

%The goal is to forecast and predict the value 0 using training instances
%One must select a training instance with length > than the length of the testing instance, here >258
%for instance model 3 is not applicable since the training instance is of length 206 (length(B{3}))
%Let considers model 2, with instance 2 of length 269:
model=2; figure, hold on, plot(B{model},'r'), plot(A{1}(:,model)), 
y = B{2}(length(A{1}(:,model))+1:end);
plot((1:length(y))+length(A{1}(:,model)),y,'k*'), title('testing data 1')
legend(['viewed by model ' num2str(model)],['forecast all using model ' num2str(model)],'forecast end only')

%The true value is given in file RUL_FD0002.txt, line "model"
%You can do similarly for other lines (other testing instances) and other datasets.

