% This file is not runnable by itself:
% you should first read and run "evaluate_similarity_polygon_rulclipper.m" until the end of the file
% to understand this content. 
% The present can then be used to evaluate the algorithm on all testing data
% and retrieve results of the paper
% 

clc, SS=0; lesRul=[];

for it_donnee_test=1:length(responses_models), % In this example, i use only one test
    % you can change this index to go over the whole testing datasets
    % below are the responses of the model that i provide and loaded in the previous .mat and which have been
    % calculated using the method presented in the paper (method of T. Wang with some modifications)
    % "A" below contains N lines, one for each datapoint in the testing
    % instance, and M columns, one for each model
    A = responses_models{it_donnee_test};
    % So the method goes along the columns and try to find the best model
    % for that it uses computational elementary operations based on
    % intersections of polygons
    lesJ = 1:size(A,2); % just a trick i used sometimes for some tests
    dis_mod2=zeros(length(lesJ),1); % will contain de similarity between test and train
    %%%%%%%%%%%%%%%%%%%%%%%%
    %% Parameters specific to CMAPSS datasets
    % Weights on subpolygons: You can see the HI as a three part signal: Part 1 remains steady,
    % Part 2 Begins to decrease, Part 3 : Decrease for sure. The weights below may be used to give
    % more weight to some parts
    size_parts_1_and_3 = 15; % samples
    poids_global = 2; % weight for part 1 2 3
    poids_fin       = 2;% weight for part 3
    poids_debut  = 1;%  weight for part 1
    % To estimate the initial offset
    offset_beginning = 50; % samples
    % Max RUL threshold, rule 3 in paper
    NMAX = 135;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% BEGIN ALGO
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j=lesJ% Predictions
        if FIGPOL, disp('---------------')
            disp(['Modele ' num2str(j)])
        end
        d1 = A(:,j);
        ladonnee=d1;
        % This is loaded in the .mat, it represents a 1D health indicator
        % signal of length T and corresponds to one model
        d2 = data_train_modele_morceaux{j};
        % Evaluate the trend at the beginning
        d=min(length(d1),length(d2));
        m1=1-mean(d1(1:min(offset_beginning,round(d/4))));
        m2=1-mean(d2(1:min(offset_beginning,round(d/4))));
        % detrend
        d1 = d1 + m1;
        d2 = d2 + m2;
        % This is for considering the three parts in the health indicators
        d=min(length(d1),length(d2));
        
        d=min(length(d1),length(d2));
        TIOdebut = fix(d/size_parts_1_and_3); % a few points before
        TIOfin = fix(d/size_parts_1_and_3);% a few points at the end
        % ensure that the polygon has three vertices
        TIOdebut = max(TIOdebut, 3);
        TIOfin = max(TIOfin, 3);
        
        % compute envelops
        [e_inf_model, e_sup_model] = enveloppe_inf_sup_signal(d2(1:d));
        [e_inf_test, e_sup_test] = enveloppe_inf_sup_signal(d1(1:d));
        if ~isempty(find(e_inf_test > e_sup_test)),
            error('Computation of envelop failed!'), end
        if FIGPOL
            figure,plot(e_sup_model,'g'), hold on, plot(e_inf_model,'g')
            hold on, plot(e_sup_test,'b'), hold on, plot(e_inf_test,'b')
            title('Envelop sup and inf of the testing instance')
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%
        % Now compute the polygon
        %%%%%%%%%%%%%%%%%%%%%%%%
        % Model global
        x1=[1:d]';            x2=[d:-1:1]';
        y1=e_sup_model(1:d);% dessous : prend debut et retourne
        y2=e_inf_model;     y2=y2(1:d);     y2=fliplr(y2')';
        lon1=[x1(TIOdebut+1:end-TIOfin);x2(TIOfin+1:end-TIOdebut)];
        lat1=[y1(TIOdebut+1:end-TIOfin);y2(TIOfin+1:end-TIOdebut)];
        % compute area between test and train
        aire_model = polyarea(lon1,lat1);
        % model at the end
        lon1surlafin=[x1(end-TIOfin+1:end);x2(1:TIOfin)];
        lat1surlafin=[y1(end-TIOfin+1:end);y2(1:TIOfin)];
        aire_model_surlafin = polyarea(lon1surlafin,lat1surlafin);
        % model at the beginning
        lon1surledebut=[x1(1:TIOdebut);x2(end-TIOdebut+1:end)];
        lat1surledebut=[y1(1:TIOdebut);y2(end-TIOdebut+1:end)];
        aire_model_surledebut = polyarea(lon1surledebut,lat1surledebut);
        %%%%%%%%%%%%%%%%%%%%%%%%
        % predicted
        x1=[1:d]';             x2=[d:-1:1]';
        y1=e_sup_test(1:d);
        y2=e_inf_test;     y2=y2(1:d);     y2=fliplr(y2')';% retourne
        lon2=[x1(TIOdebut+1:end-TIOfin);x2(TIOfin+1:end-TIOdebut)];
        lat2=[y1(TIOdebut+1:end-TIOfin);y2(TIOfin+1:end-TIOdebut)];
        % area
        aire_pred = polyarea(lon2,lat2);
        % pred at the end
        lon2surlafin=[x1(end-TIOfin+1:end);x2(1:TIOfin)];
        lat2surlafin=[y1(end-TIOfin+1:end);y2(1:TIOfin)];
        aire_pred_surlafin = polyarea(lon2surlafin,lat2surlafin);
        % pred at the beginning
        lon2surledebut=[x1(1:TIOdebut);x2(end-TIOdebut+1:end)];
        lat2surledebut=[y1(1:TIOdebut);y2(end-TIOdebut+1:end)];
        aire_pred_surledebut = polyarea(lon2surledebut,lat2surledebut);
        
        % Display
        if FIGPOL
            figure(300),close(gcf),figure(300),
            plot(d2), hold on
            fill(lon1,lat1,'c')
            hold on, fill(lon2,lat2,'y')
            %fill(lon1surlafin,lat1surlafin,'m')
            %fill(lon1surledebut,lat1surledebut,'g')
            plot((rul1(it_donnee_test)+size(ladonnee,1))*ones(length([min(d2):max(d2)]),1),[min(d2):max(d2)]','k-','LineWidth',3)
            title('Part 2 (c), global prediction (y) real RUL (k)')
            figure(301),close(gcf),figure(301),
            plot(d2), hold on
            %fill(lon1,lat1,'c')
            hold on, fill(lon2surlafin,lat2surlafin,'y')
            fill(lon1surlafin,lat1surlafin,'c')
            %fill(lon1surledebut,lat1surledebut,'g')
            plot((rul1(it_donnee_test)+size(ladonnee,1))*ones(length([min(d2):max(d2)]),1),[min(d2):max(d2)]','k-','LineWidth',3)
            title('Part 3 (end) (c), global prediction (y) real RUL (k)')
            figure(302),close(gcf),figure(302),
            plot(d2), hold on
            %fill(lon1,lat1,'c')
            %hold on, fill(lon2,lat2,'y')
            fill(lon2surledebut,lat2surledebut,'y')
            fill(lon1surledebut,lat1surledebut,'c')
            plot((rul1(it_donnee_test)+size(ladonnee,1))*ones(length([min(d2):max(d2)]),1),[min(d2):max(d2)]','k-','LineWidth',3)
            title('Part 1 (beginning) (c), global prediction (y) real RUL (k)')
        end
        % INTERSECTION between predicted and model
        [loni,lati] = polybool('intersection',lon1,lat1,lon2,lat2);
        % compute area
        loni=[nan ; loni ; nan];
        lati=[nan ; lati ; nan];
        f=find(isnan(loni));
        k=2;
        if FIGPOL, figure(303),close(gcf),figure(303), end
        aire_int=0;
        if length(loni)>0 && length(f)==0
            f=length(loni);
        end
        for i=1:length(f)
            x=loni(k:f(i)-1);
            y=lati(k:f(i)-1);
            aire_int = aire_int+polyarea(x,y);
            if FIGPOL
                plot(d2), hold on
                plot((rul1(it_donnee_test)+size(ladonnee,1))*ones(length([min(d2):max(d2)]),1),[min(d2):max(d2)]','k-','LineWidth',3)
                hold on, fill(x,y,'r')
                title('Area of inteserction between predicted and model (r), true RUL (k)')
            end
            k=f(i)+1;
        end
        clear i f k x y
        % degree of intersection
        aire_int_nonnorm = aire_int;
        %%%%%%%%%%%%%%%%%%%%%
        if FIGPOL, disp(['intersect full  ' num2str([aire_int_nonnorm])]), end
        %%%%%%%%%%%%%%%%%%%%%
        % INTERSECTION at the end
        [lonisurlafin,latisurlafin] = polybool('intersection',lon1surlafin,lat1surlafin,lon2surlafin,lat2surlafin);
        % area
        lonisurlafin=[nan ; lonisurlafin ; nan];
        latisurlafin=[nan ; latisurlafin ; nan];
        f=find(isnan(lonisurlafin));
        k=2;
        aire_int_surlafin=0;
        if length(lonisurlafin)>0 && length(f)==0
            f=length(lonisurlafin);
        end
        for i=1:length(f)
            x=lonisurlafin(k:f(i)-1);
            y=latisurlafin(k:f(i)-1);
            aire_int_surlafin = aire_int_surlafin + polyarea(x,y);
            k=f(i)+1;
        end, clear f k i y x
        aire_int_surlafin_nonnorm = aire_int_surlafin;
        %%%%%%%%%%%%%%%%%%%%%
        if FIGPOL, disp(['intersect end   ' num2str([aire_int_surlafin_nonnorm])]), end
        %%%%%%%%%%%%%%%%%%%%%
        % INTERSECTION beginning
        [lonisurledebut,latisurledebut] = polybool('intersection',lon1surledebut,lat1surledebut,lon2surledebut,lat2surledebut);
        % area
        lonisurledebut=[nan ; lonisurledebut ; nan];
        latisurledebut=[nan ; latisurledebut ; nan];
        f=find(isnan(lonisurledebut));
        k=2;
        aire_int_surledebut=0;
        if length(lonisurledebut)>0 && length(f)==0
            f=length(lonisurledebut);
        end
        for i=1:length(f)
            x=lonisurledebut(k:f(i)-1);
            y=latisurledebut(k:f(i)-1);
            aire_int_surledebut = aire_int_surledebut + polyarea(x,y);
            k=f(i)+1;
        end, clear f k i x y
        aire_int_surledebut_nonnorm = aire_int_surledebut;
        %%%%%%%%%%%%%%%%%%%%%
        if FIGPOL, disp(['intersect beginning ' num2str([aire_int_surledebut_nonnorm])]), end
        %%%%%%%%%%%%%%%%%%%%%
        % Precision = Predicted INTER Modele / Pred
        if FIGPOL, disp(sprintf('Mesures \t Global \t Debut \t Fin')),end
        precis_val = aire_int_nonnorm / aire_pred;
        precis_val_surlafin = aire_int_surlafin_nonnorm / aire_pred_surlafin;
        precis_val_surledebut = aire_int_surledebut_nonnorm / aire_pred_surledebut;
        if FIGPOL, disp(sprintf('Precision \t %f \t %f \t %f', precis_val, precis_val_surledebut, precis_val_surlafin)), end
        % Modele = Predicted INTER Modele / Modele
        rappel_val = aire_int_nonnorm / aire_model;
        rappel_val_surlafin = aire_int_surlafin_nonnorm / aire_model_surlafin;
        rappel_val_surledebut = aire_int_surledebut_nonnorm / aire_model_surledebut;
        if FIGPOL, disp(sprintf('Recall \t %f \t %f \t %f', rappel_val, rappel_val_surledebut, rappel_val_surlafin)), end
        % Fmeasure
        FBeta_global = fbetamesure(precis_val, rappel_val, 1);
        FBeta_fin = fbetamesure(precis_val_surlafin, rappel_val_surlafin, 1);
        FBeta_debut = fbetamesure(precis_val_surledebut, rappel_val_surledebut, 1);
        if FIGPOL, disp(sprintf('Fmesures \t %f \t %f \t %f', FBeta_global, FBeta_debut, FBeta_fin)),
        end
        % Similarity
        s=poids_global+poids_fin+poids_debut;
        poids_global=poids_global/s; poids_fin=poids_fin/s; poids_debut=poids_debut/s;
        FBeta_fusion = FBeta_global*poids_global + FBeta_debut*poids_debut + FBeta_fin*poids_fin;
        if isnan(FBeta_fusion), FBeta_fusion=0; end
        % Below is a rule explained in the paper (RULE 3), especially developed for CMAPSS
        d1b=smooth(d1(1:d),15); d2b=smooth(d2(1:d),15);
        f = length(d1b):-1:length(d1b)-35; f(find(f<=0))=[];
        if length(f)>0
            cond1 = length(d1) <= length(d2)/2; % test "court" wrt modele
            if cond1
                F = d1b(f) < d2b(f); c=0;
                for l=2:length(F), if F(l)==F(l-1), c=c+F(l); else c=0.8*c+F(l); end, end
                if c >= 25 % test au dessous => on va surestimer
                    FBeta_fusion = 0;
                else
                    F = d1b(f) > d2b(f); c=0;
                    for l=2:length(F), if F(l)==F(l-1), c=c+F(l); else c=0.8*c+F(l); end, end
                    if c >= 25
                        FBeta_fusion = 0;
                    end
                end
            end
        end
        if FIGPOL, disp(sprintf('Fmesure Fusion \t %f \n (estimated RUL = %d if this instance is selected, vs true RUL %d)', FBeta_fusion, (size(data_train_modele_morceaux{j},1)-length(ladonnee)), rul1(it_donnee_test))), end
        if FBeta_fusion>1
            error('??')
        end
        dis_mod2(j,1) = FBeta_fusion;
        if FIGPOL, disp('Tapez une touche...'), pause, end
        clear x1 x2 y1 y2 k i lat1 lat2 lon1 lon2 lati loni
        close all
    end
    disp('From here all models have been applied for the current testing instance')
    disp('Press enter to continue: compute RUL and make comparison')
    % dis_mod2 contains all Fmeasures
    % and 0=>problem in general => TO BE MANAGED
    if FIGPOL, figure,plot(dis_mod2),title('The higher the better'), end
    % sort by decreasing order
    [a b]=sort(dis_mod2,'descend');
    % b([1 2 3 ...]) are the best matches
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % compute RULs
    clear r
    for i=1:length(a)%data_train)
        r(i) = size(data_train_modele_morceaux{b(i)},1) - size(A,1);%size(data_test{it_donnee_test},1);
    end
    r(find(r<0))=10; % too short, useless...
    f=find(a<eps);
    if length(f)==length(a), error('??'), end
    
    b(f)=[];    a(f)=[];    r(f)=[];
    % RUL MAX (dependent on the dataset, to be adapted)
    % Rule 1 explained in the paper
    warning('Adapt MAXRUL threshold according to dataset, please see the paper tables 5-7')
    disp(sprintf('ATTENTION SEUIL A %d',NMAX))
    f=find(r>NMAX);  r(f)=NMAX;
    % just a display
    for i=1:min(10,length(r))
        if (r(i)<10 && b(i)<100) || (b(i)<10 && r(i)<100)
            disp(sprintf('%d (%d) \t \t %d \t %f',r(i), b(i), rul1(it_donnee_test), a(i)))
        else
            disp(sprintf('%d (%d)  \t %d \t %f',r(i), b(i), rul1(it_donnee_test), a(i)))
        end
    end
    % FUSION using the RUL in Table 5
    warning('Adapt the fusion according to the dataset, see paper tables 5-7')
    q=min(length(r),11);
    r5 = [r(1:q)]; f = find(a(1:q)>0.5);
    if ~isempty(f), r5=r5(f); b5=b(f); a5=b(f);
    else  f = find(a(1:q)>0.5);
        if ~isempty(f), r5=r5(f); b5=b(f); a5=b(f);
        end
    end
    
    % Finally the rule given in tables 5-6
    if datasetNumber==1
        % for DATASET 1, first line:
        RULestimated = 0.5*(0.9*min(r5)+0.1*max(r5)) + ...
            0.5*[ sum( (a(1:min(7,length(b)))' .* r(1:min(7,length(b)))) ) / sum(a(1:min(7,length(b)))) ]
        
    elseif datasetNumber==2
        % table 6, first line:
        % The first part of the rule
        % discrete sequences
        Z=[]; P=[];
        for i=1:min(8,length(b)),
            S=length(data_train_modele_morceaux{b(i)})-15; % start of state 2
            S = S - size(A,1);
            if S>0, Z=[Z S]; end
        end
        Z(find(Z>NMAX))=NMAX;
        
        r6 = [r(1:q) Z];
        
        [a6 b6 c6]=calc_quartile_up_low_med(r6);
        f=find(r6>(a6 + 2*(c6-a6)));
        if length(f)<length(r), r6(f)=[]; end
        f=find(r6<(a6 - 3*(a6-b6)));
        if length(f)<length(r), r6(f)=[]; end
        
        % the second part of the rule
        % moving weighted average with removal outliers"
        q=min(length(r),15);
        r2=r(1:q); a2=a(1:q); b2=b(1:q);
        if mod(length(r2),2)==0,  r2=r(1:q-1); a2=a(1:q-1); b2=b(1:q-1); end
        [a1 b1 c1]=calc_quartile_up_low_med(r2);
        f1=find(r2<b1); f2=find(r2>c1);
        r2(union(f1,f2))=[]; a2(union(f1,f2))=[]; b2(union(f1,f2))=[];
        mw3OR = sum( (a2(1:min(3,length(b2)))' .* r2(1:min(3,length(b2)))) ) / sum(a2(1:min(3,length(b2))));
        
        % finally
        RULestimated = 0.5*mw3OR + 0.5*median(r6)
    
    elseif datasetNumber==3, error('to be done')
    elseif datasetNumber==4, error('to be done')
    else error('??')
    end
    
    RULestimated = min(RULestimated, NMAX);
    
    warning('you should implement other rules for the other datasets')
    % Plot results
   
    figure,hold on
    h=plot(A(:,b(1:q)),'m.');
    s={'inputs'}; % for legend
    for i=1:q
        h=[h ; plot(data_train_modele_morceaux{b(i)},'Color',rand(1,3))];
        s{end+1}=sprintf('model %d',b(i));
    end
    h=[h ; line(size(A,1)+ones(1,10)*rul1(it_donnee_test),linspace(-0.1,1.3,10),'linewidth',3)];
    s{end+1}=sprintf('True RUL %d',rul1(it_donnee_test));
    h=[h ; line(size(A,1)+ones(1,10)*RULestimated,linspace(-0.1,1.3,10),'linewidth',3,'Color',[1 0 0])];
    s{end+1}=sprintf('Estimated RUL %d',round(RULestimated));
    legend(h(q:end),s)
    % Error function
    [s, ~,~,~,~,~,ac]=prognostics_metrics(rul1(it_donnee_test), RULestimated,13,10);
    SS=SS+s
    lesRul = [lesRul ; RULestimated];
    it_donnee_test
end

[S FPR FNR MAPE MAE MSE Acc] = prognostics_metrics(rul1, lesRul,13,10);
sum(S), Acc, 
FPR, FNR, MAPE, MAE, MSE

