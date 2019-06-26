function [responses_models, data_train, data_train_modele_morceaux, directoryToDataset] = ...
    health_indicators_estimation_cmapss(datasetNumber, sensorIndex, nbRegimes, globalComputation)
%% Estimate the health indicators for CMAPSS turbofan datasets using the method [1]
% Warning: Requires the CMAPSS datasets to be downloaded on NASA PCOE. Those datasets
% and benchmarking are fully described in [2]. Download here:
% http://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/
% in section called: "Turbofan Engine Degradation Simulation Data Set",
% those files contain RUL_FD000x.txt which contains the RULs for x=1,2,3,4 and other datasets files.
% Put everything in a folder. Do not change their names. 
%
% Usage:
% [A B C] = health_indicators_estimation_cmapss(datasetNumber, sensorIndex, nbRegimes, globalComputation);
% estimates HI on dataset "datasetNumber" (an integrer) using column "sensorIndex"
% (vector of integer), using "nbRegimes" (an integer, generally 1 or 6, see the datasets decsription) 
% and applying global or local computation ("globalComputation" is false or true). 
%
% Example: 
% [A B C] = health_indicators_estimation_cmapss(2, [7 8 9 10 12 14 16 17 20 25 26], 6, true);
% estimate HI on dataset 2 using column sensors [7 8 9 10 12 14 16 17 20 25 26]
% with 6 regimes and using the global computation. Local computation could be better
% for dataset #2 or #4, but global can be interested to tackle. It is used in a method described in [1]:
%
% Outputs: 
% A=is a cell where response{i}(1:t,j) is the application of modele j=1...Nb_training_instances 
% on TESTING data i with length 1:t.
%
% B=is a cell of health indicators of TRAINING data computed with the GLOBAL method (see [1])
%
% C=is a cell of health indicators of TRAINING data computed with the LOCAL method (see [1]). It is 
% computed for dataset with operating conditions. So C is empty for dataset 1 or 3 since with 
% one operating conditions.
%
% directoryToDataset is the path to the datasets
%
% Example, cont.: 
% %this is the response of model 3 for the testing data 1 (of length 258) in dataset 2 obtained above
% figure,plot(A{1}(:,3)),
% %The goal is to forecast and predict the value 0 using training instances
% %One must select a training instance with length > than the length of the testing instance, here >258
% %for instance model 3 is not applicable since the training instance is of length 206 (length(B{3}))
% %Let considers model 2, with instance 2 of length 269:
% model=2; figure, hold on, plot(B{model},'r'), plot(A{1}(:,model)), 
% y = B{2}(length(A{1}(:,model))+1:end);
% plot((1:length(y))+length(A{1}(:,model)),y,'k*'), title('testing data 1')
% legend(['viewed by model ' num2str(model)],['forecast all using model ' num2str(model)],'forecast end only')
% %The true value is given in file RUL_FD0002.txt, line "model"
% %You can do similarly for other lines (other testing instances) and other datasets.
% 
%
% This code runs for datasets 1 to 4, please adapt it as necessary for an
% application on the PHM data challenge as i did in the paper. 
%
% In case of problems, please contact me.
% Some warnings may appear due to regression with few points, does not
% matter so much... could be improved probably
%
% Author: Emmanuel Ramasso (emmanuel.ramasso@femto-st.fr), Dec. 2015
%
% Reference
%
% [1] E. Ramasso, Investigating computational geometry for failure prognostics,
% International Journal on Prognostics and Health Management, 5(5):1-18, 2014.
%
% [2] E. Ramasso and A. Saxena, Performance Benchmarking and Analysis of Prognostic Methods 
% for CMAPSS Datasets. International Journal on Prognostics and Health Management, 5(2):1-15, Dec. 2014. 
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Suppose you have download CMAPSS datasets at 
% http://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/
% in section called: "Turbofan Engine Degradation Simulation Data Set",
% those files contain RUL_FD000x.txt which contains the RULs for x=1,2,3,4 and other datasets files.
% no change their name

R = uigetdir('../../../6_BENCHMARK/CMAPSSData','Choose directory containing CMAPSS data');
directoryToDataset = R;

PL_FIG = 0;

% the data contain 26 cols with unit nb, time, op set 1-3 then sensors readings
disp('load file...')
if any(datasetNumber==[1 2 3 4])
    train1=dlmread([R '/train_FD00' num2str(datasetNumber) '.txt'],' ');
    train1=train1(:,1:26);
    
    test1=dlmread([R '/test_FD00' num2str(datasetNumber) '.txt'],' ');
    test1=test1(:,1:26);
    
    rul1 = dlmread([R '/RUL_FD00' num2str(datasetNumber) '.txt']);
else error('dataset: bad number ??')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARAM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% see the documentation of CMAPSS
% contain operating modes 
LES_MODES = [3 4 5];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build HI : Global method => generate data_train{i}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

uniq=unique(train1(:,1));
data_train = cell(length(uniq),1);
modele_lineaire_global = cell(length(uniq),1);

l=[];
for i=1:length(uniq)
    
    ff=find(train1(:,1)==uniq(i));
    x=train1(ff,:);
    
    % ou bien modele sur LISSE ?
    data_train{i} = x(:,sensorIndex);
    
    y=[size(data_train{i},1):-1:1]'-1;
    f=95*length(y)/100; f=f/max(y); f=-f/log(0.05);
    y=y/max(y);
    y=1-exp(-y/f);
    C=round(min([1.2*length(sensorIndex), 5*length(y)/100])); % just to avoid to many columns for few data
    y(1:C)=1;
    y(end-C+1:end)=0;
    
    %figure,plot(y)
    
    brob=robustfit(data_train{i},y);
    
    modele_lineaire_global{i}=brob;
    
    data_train{i} = sum(repmat(brob',size(data_train{i},1),1) .* [ones(size(data_train{i},1),1) data_train{i}],2);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functioning modes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if datasetNumber==4 || datasetNumber==2
    centres = ...
        [42.0030 0.8405 100 ;
        35.0030 0.8405 100 ;
        0.0015   0.0005 100 ;
        20.003   0.7005 100 ;
        25.0031 0.6205 60   ;
        10.003   0.2505 100 ]
elseif datasetNumber==1 || datasetNumber==3
    centres = [...
        0        0     100 ;
        0        0     100 ;
        0        0     100 ;
        0        0     100 ;
        0        0     100 ;
        0        0     100 ]
end

k=size(centres,1);
n=size(train1,1);
D = zeros(k,n);
for l = 1:k
    z=(train1(:,LES_MODES)- ones(n, 1)*centres(l, :)).^2;
    D(l,:)=(sum(z,2))';
end
[Dmin,regimes]=min(D);
regimes=regimes'; clear n k Dmin D l z

% or works also:
%[regimes centres]=kmean(train1(:,LES_MODES),nbRegimes,10);

% UNIT NUMBER en 1
les_modes_train_2_fonct = cell(length(uniq),1);
uniq=unique(train1(:,1));
for i=1:length(uniq)
    ff=find(train1(:,1)==uniq(i));
    les_modes_train_2_fonct{i} = regimes(ff);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% build HI : Local method => generate data_train_modele_morceaux{i}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% build one model for each mode for each trajectory
uniq=unique(train1(:,1));
modeles_lineaires_morceau = cell(length(uniq),1);
data_train_modele_morceaux = cell(length(uniq),1);

if ~globalComputation %local
    
    for i=1:length(uniq)
        
        ff=find(train1(:,1)==uniq(i));
        x=train1(ff,:);
        
        d = x(:,sensorIndex);
        
        y=[size(x,1):-1:1]'-1;
        f=95*length(y)/100; f=f/max(y); f=-f/log(0.05);
        y=y/max(y);
        y=1-exp(-y/f);
        C=round(min([1.2*length(sensorIndex), 5*length(y)/100])); % just to avoid to many columns for few data
        y(1:C)=1;
        y(end-C+1:end)=0;
        
        HIDX2 = zeros(size(d,1),1);
        for j=1:nbRegimes
            f=find(les_modes_train_2_fonct{i}==j);
            if isempty(f)
                modeles_lineaires_morceau{i}{j}=[];
                disp(sprintf('Morceau vide %d %d',i,j))
            else
                try
                    brob=robustfit(d(f,:),y(f));%,'cauchy');
                    %brob=regress(y(f),[ones(length(f),1) d(f,:)]);
                catch ME
                    if strcmp(ME.identifier,'stats:robustfit:NotEnoughData')
                        disp(sprintf('Donnee %d - modele %d --> ROBUSTFIT manque de pts => Essaie regress!',i,j));
                        
                        try
                            brob=regress(y(f),[ones(length(f),1) d(f,:)]);
                        catch ME
                            if strcmp(ME.identifier,'stats:robustfit:NotEnoughData')
                                disp(sprintf('Donnee %d - modele %d --> ROBUSTFIT et REGRESS manque de pts => EMPTY modele!',i,j));
                                return;
                                %brob=regress(y(f),[ones(length(f),1) d(f,:)]);
                                %else error('??')
                            end
                        end
                        
                    end
                end
                modeles_lineaires_morceau{i}{j}=brob;
                HIDX2(f) = sum(repmat(brob',size(d(f,:),1),1) .* [ones(size(d(f,:),1),1) d(f,:)],2);
            end
        end
        
        data_train_modele_morceaux{i} = HIDX2;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% for test, prepare "features modes" + data
uniq2=unique(test1(:,1));
data_test = cell(length(uniq),1);
modes_test = cell(length(uniq),1);
for i=1:length(uniq2)
    data_test{i}=test1(find(test1(:,1)==uniq2(i)),:);
    d = data_test{i}(:,LES_MODES);
    modes_test{i} = d;
    data_test{i} = data_test{i}(:,sensorIndex);
end

%%%%%%%%%%%%%%%%%%
clear train1 uniq uniq2 x y regimes les_modes_train_2_fonct ff f dd d
%%%%%%%%%%%%%%%%%%

% Apply RULCLIPPER algorithm
responses_models = cell(length(rul1),1);

for it_donnee_test = 1:length(rul1)
    
    disp('################################')
    disp(sprintf('%d/%d (#%d)',it_donnee_test,length(rul1),datasetNumber))
    disp('################################')
    
    
    ladonnee=data_test{it_donnee_test};
    
    % trouve les regimes passes pour la donnee courante
    k=size(centres,1);
    n=size(ladonnee,1);

    % find the mode
    D = zeros(k,n);
    for l = 1:k
        z=(modes_test{it_donnee_test} - ones(n, 1)*centres(l, :)).^2;
        D(l,:)=(sum(z,2))';
    end
    [Dmin,klas]=min(D);
    
    if PL_FIG
        figure(100),plot(klas),title('Segmentation de la donnee courante')
    end
    
    A=zeros(length(klas),length(data_train));
    
    % Apply model
    if globalComputation
        
        for j=1:length(data_train)
            modele_courant = modele_lineaire_global{j};
            for k=1:n
                if ~isempty(modele_courant)
                    A(k,j) = sum(modele_courant' .* [1 ladonnee(k,:)],2);
                end
            end
        end
        
    else
        
        for j=1:length(data_train)
            for k=1:length(klas)
                modes_courant = klas(k);
                modele_courant = modeles_lineaires_morceau{j}{modes_courant};
                if ~isempty(modele_courant)
                    A(k,j) = sum(modele_courant' .* [1 ladonnee(k,:)],2);
                end
            end
        end
        
    end
    
    responses_models{it_donnee_test} = A;
        
end
