% CODE FOR MULTIVARIATE ANALYSIS OF BEHAVIORAL VARIABLES
% 
% Used in the article: Oliveira-Junior et al. Multidimensional behavioral 
%   profiles associated with resilience and susceptibility after 
%   inescapable stress. Scientific Reports (2024).
% 
% OBS: This script was designed to run all analyses and generate all 
%   results and figures in a single run using the data from the file 
%   'Data_Table.xlsx', available at: https://osf.io/mgknc. You may run the 
%   entire script or each section separately in order. To run this script
%   you need to add the 'utilities' folder with custom scripts in the path.
%   Note that for analyses based on randomly selecting partitioned data, 
%   the results may slightly vary between runs.
% 
% Authors: Danilo Benette Marques, Benedito Alves de Oliveira Júnior,
%   Rafael Naime Ruggiero (2019-2024).
% Acknowledgments: This script was primarily created by Danilo Benette
%   Marques and was significantly developed, thoroughly tested, and 
%   discussed with Benedito Alves de Oliveira Júnior and Rafael Naime 
%   Ruggiero.
% 
% Institution: Department of Neurosciences and Behavioral Sciences, 
%   Ribeirão Preto Medical School, University of São Paulo.

%% Load and set data for multivariate analysis
clear all
close all
clc

%Set path or get 'Data_Table.xlsx' file available at: https://osf.io/mgknc
filename = uigetfile('*.*'); %e.g.: filename = '...\Data_Table.xlsx';

%Read table
TBL = readtable(filename);

DATA = table2array(TBL(:,3:end));
VARLABELS = TBL.Properties.VariableNames(3:end);

% %% Rename some relevant variable labels
VARLABELS{1} = 'OF (Block 1, Dist.)';
    VARLABELS{2} = 'OF (Block 2, Dist.)';
    VARLABELS{3} = 'OF (Block 3, Dist.)';
    VARLABELS{4} = 'OF (Block 4, Dist.)';
    VARLABELS{5} = 'OF (Total Dist.)';
    VARLABELS{7} = 'OF (Time in Center)';
    VARLABELS{8} = 'OF (Rearing Events)';
    VARLABELS{13} = 'FST (Train, Immob.)';
    VARLABELS{17} = 'FST (Train, Climb.)';
    VARLABELS{21} = 'FST (Train, Swim.)';
    VARLABELS{30} = 'FST (Immob.)';
    VARLABELS{36} = 'FST (Climb.)';
    VARLABELS{42} = 'FST (Swim.)';
    VARLABELS{46} = 'SIT (Cage)';
    VARLABELS{47} = 'SIT (Subject)';
    VARLABELS{50} = 'SIT (Soc. Pref. Ratio)';
    VARLABELS{53} = 'EPM (Open Arms)';
    VARLABELS{54} = 'EPM (Closed Arms)';
    VARLABELS{56} = 'EPM (Risk Assessment)';
    VARLABELS{81} = 'SPT (D1, Water)';
    VARLABELS{82} = 'SPT (D1, Sucrose)';
    VARLABELS{84} = 'SPT (D2, Water)';
    VARLABELS{85} = 'SPT (D2, Sucrose)';
    VARLABELS{87} = 'SPT (D3, Water)';
    VARLABELS{88} = 'SPT (D3, Sucrose)';
    VARLABELS{93} = 'SPT (D1, Suc. Pref. %)';
    VARLABELS{94} = 'SPT (D2, Suc. Pref. %)';
    VARLABELS{95} = 'SPT (D3, Suc. Pref. %)';
    VARLABELS{96} = 'SPT (Suc. Pref. %)';
    VARLABELS{97} = 'SB (Escape Failures)';
    VARLABELS{98} = 'SB (1-5 FR1, Latency)';
    VARLABELS{99} = 'SB (5-25 FR1, Latency)';
    VARLABELS{100} = 'SB (5-25 FR2, Latency)';
    
%Test labels
testlabels = {'OF','FST','SIT','EPM','NOR','PPI','SPT','SB'};
testcolor = [1 .5 .25; 0.25 .75 1; .8 0 0; .5 0 1; .5 1 .5 ; 1 .5 .5; 1 .9 0; .75 .75 .75];

%Identify groups NS and IS
idx_group = table2array(TBL(:,2)); %NS(0), IS(1)

% %% Exclusion by technical issues
idx_exclude = [12 25]; 
    DATA(idx_exclude,:) = [];
    idx_group(idx_exclude,:) = [];
    
% %% Identify helpless (H) and non-helpless (NH)
rhdata = DATA(:,[97 100]); %shuttle box escape performance measures
[idx_rh] = kmeans(zscore(rhdata,1),2,'replicates',1000,'maxiter',100,'display','off');
idx_rh = sortk(idx_rh,rhdata)-1; %NH(0), H(1)
clear rhdata

% %% Select variables of interest and define datasets
% Dataset 1: relevant for depression/anxiety
% Indices of variables
idx_vars_relevant = ... 
    [1 2 3 4   7 8 ... %OF
    13 17 21 30 36 42 ... %FST
    46 47 50 ... %SIT
    53 54 56 ... %EPM
    81 82 84 85 93 94 95 ... %SPT
    97 98 99 100]; %SB

% Indices of resilience- and susceptibility-related variables
idx_vars_res_relevant = []; 
    idx_vars_res_relevant([1 2 3 4 6 ...
        7 10 13 ...
        17 18 ...
        19 21 ...
        26 27 28 29]) = -1;
    idx_vars_res_relevant([5 ...
        8 9 11 12 ...
        14 15 ...
        16 ...
        20 22 23 24 25]) = 1;  

% Indices of variables' test
idx_vars_test_relevant = []; 
    idx_vars_test_relevant(1:6) = 1; %OF
    idx_vars_test_relevant(7:12) = 2; %FST
    idx_vars_test_relevant(13:15) = 3; %SIT
    idx_vars_test_relevant(16:18) = 4; %EPM
    idx_vars_test_relevant([]) = 5; %NOR
    idx_vars_test_relevant([]) = 6; %PPI
    idx_vars_test_relevant(19:25) = 7; %SPT
    idx_vars_test_relevant(26:29) = 8; %EP

%Dataset 2: top relevant for depression/anxiety
idx_vars_top = [5 30 50 53 96 97]; %indices of variables
    idx_vars_res_top = [-1 -1 1 1 1 -1]; %resilience(1),susceptibility(-1)
    idx_vars_test_top = [1 2 3 4 7 8]; %test
   
%% FEATURE CLUSTERING
%For feature clustering analysis, we used the 29 relevant variables
idx_vars = idx_vars_relevant;
    data = DATA(:,idx_vars);
    varlabels = VARLABELS(idx_vars);
idx_vars_res = idx_vars_res_relevant;
idx_vars_test = idx_vars_test_relevant;

%Normalize
data = normalize(data,1,'zscore');

% %% Shuffled data
datashuffle = shuffle(data,1); %shuffle obs
for idx_rat = 1:size(datashuffle,1)
    datashuffle(idx_rat,:) = shuffle(datashuffle(idx_rat,:),2); %shuffle vars
end

%% Code for Figure 3A-B
% Hierarchical clustering (HC) of behavioral variables 
% Tests relationship with resilience- and susceptibility-related variables

%Hierarchical clustering
k = 2; %number of clusters (resilience vs. susceptibility?)

distfun = @(i,j) pdist2(i,j); %"classic" Euclidean distance
hcvarsmethod='ward'; %Ward's method

Zvars = linkage(data',hcvarsmethod,distfun); %tree clustering
    Dvars = pdist(data'); %distance distributions
    sortvars = optimalleaforder(Zvars,Dvars); %indices or dendrogram-sorted variables (used throughout this script)
    Tvars = cluster(Zvars,'maxclust',k); %cluster by defined number of clusters
    COPHvars = cophenet(Zvars,Dvars) %cophenetic coefficient

%Plot dendrogram
figure('color','w'),
    [H] = dendrogram(Zvars,0,'orientation','top','reorder',sortvars,'colorthreshold',Zvars(end,3));
    ylabel('Dissimilarity')
    set(H,'linewidth',1.5,'color','k')
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    pause(0)

%Chi-squared test clusters x resilience/susceptibility
[tbl,chi2,pchi2] = crosstab(Tvars,idx_vars_res); chi2, pchi2, 
    %Plot counts of clusters x resilience/susceptibility
    figure('color','w'),bar(tbl,1); ylabel('Counts'); xlabel('Cluster') 
    legend({'Resilience.','Susceptibility.'},'box','off')
    %Plot dendrogram-sorted resilience/susceptibility variables
    figure('color','w'),imagesc(idx_vars_res(sortvars)); colormap([.8 0 0 ; 0 0 .8]); set(gca,'visible','off')
    pause(0)

%% Code for Figure 3A
% Correlation matrix

%Correlation matrix
[C,pcorr] = corr(data,'type','spearman'); 
    C(find(eye(size(C)))) = 1; %make diagonal 1
    
%Plot correlation matrix
figure('color','w'),
    imagesc(C(sortvars,sortvars)); %Spearman's correlation coefficient
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    set(gca,'YTick',[1:length(idx_vars)],'YTickLabel',varlabels(sortvars),'YTickLabelRotation',0) 
    title('Correlation matrix')
    colormap(colormaprwb); caxis([-.5 .5]); c=colorbar; ylabel(c,'Correlation (r_s)'); set(c,'ytick',[-.5 0 .5])    
    axis square
    corrax1 = gca;
    pause(0)

figure('color','w'),
    imagesc(pcorr(sortvars,sortvars)); %Spearman's correlation p-value
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    set(gca,'YTick',[1:length(idx_vars)],'YTickLabel',varlabels(sortvars),'YTickLabelRotation',0) 
    title('Significance of correlations')
    colormap bone; caxis([0 0.05])    
    c=colorbar; ylabel(c,'{\it p}-value'); set(c,'ytick',[0 .05])    
    axis square
    corrax2 = gca;
    linkaxes([corrax1 corrax2],'xy')
    pause(0)

%Display which variables are significantly correlated
clear corrlabels
for ivar = 1:length(varlabels)                 
isignpcorr = find(pcorr(:,ivar)<0.05); 
isignpcorr = sortrows([isignpcorr pcorr(isignpcorr,ivar)],2); isignpcorr(:,2)=[];
if ~isempty(isignpcorr)
    disp([varlabels{ivar} ' correlates with: '])
    disp(varlabels(isignpcorr)');
end
corrlabels{1,ivar} = (varlabels(ivar));
corrlabels{2,ivar} = [(varlabels(isignpcorr))' num2cell(C(isignpcorr,ivar)) num2cell(pcorr(isignpcorr,ivar))];
end

%% Code for Figure 3C
% Permutation tests of hierarchical clustering of variables
% Tests relationship with resilience- and susceptibility-related variables

%Get original sizes
Nobs = size(data,1); idx_obs_iter = 1:Nobs; %number and indices of origial observations
Nvars = size(data,2); idx_vars_iter = 1:Nvars; %number and indices of original variables

%HC parameters
distfun = @(i,j) pdist2(i,j); %Euclidean distance
hcvarsmethod='ward'; %Ward's method 
k = 2; %number of clusters (resilience vs. susceptibility)

% %% Permutation of observations
clear Chi2 *pChi2;
for iter = 1:10000 %iteration
    
    %Permutation of observations
    Nobsiter = round(.7*Nobs); %70% of original observations
    idx_obs_iter = randperm(Nobs,Nobsiter); %random 70% indices of observations
    
    dataiter = data(idx_obs_iter,idx_vars_iter); %random selection of observations

    Z = linkage(dataiter',hcvarsmethod,distfun);
        T = cluster(Z,'maxclust',k);
    
    %Chi-squared test of iteration clusters x resilience/susceptibility
    [~,Chi2(iter),pChi2(iter)] = crosstab(T,idx_vars_res(idx_vars_iter)); 
    
    clc
end

%Permutation of observations of shuffled data
clear Chi2shuffle *pChi2shuffle;
for iter = 1:10000 %iteration
    
    %Permutation of observations
    Nobsiter = round(.7*Nobs); %70% of original observations
    idx_obs_iter = randperm(Nobs,Nobsiter); %random 70% indices of observations
    
    dataiter = datashuffle(idx_obs_iter,idx_vars_iter); %random selection of observations

    Z = linkage(dataiter',hcvarsmethod,distfun);
        T = cluster(Z,'maxclust',k);
    
    %Chi-squared test of iteration clusters x resilience/susceptibility
    [~,Chi2shuffle(iter),pChi2shuffle(iter)] = crosstab(T,idx_vars_res(idx_vars_iter)); 
    
    clc
end

%Plot histogram of p-values across iterations of partitioned observations
figure('color','w')
histogram(pChi2,0:.01:1,'normalization','probability','facecolor',[1 0 0]) %real data
hold on,histogram(pChi2shuffle,0:.01:1,'normalization','probability','facecolor',[.5 .5 .5]) %shuffled data
gridxy(.05,[],'linestyle','--','color','k')
    title('Permutation of observations')
    ylabel('Probability')
    xlabel('Chi-squared test {\it p}-value')
    legend({'','data','shuffle'},'box','off')
    set(gca,'box','off')
    pause(0)
    
probpChi2obs = 100*mean(pChi2<0.05)

% %% Permutation of variables
clear Chi2 *pChi2;
for iter = 1:10000 %iteration
    
    %Permutation of variables
    Nvarsiter = round(.7*Nvars); %70% of original variables
    idx_vars_iter = randperm(Nvars,Nvarsiter); %random 70% indices of variables
    
    dataiter = data(idx_obs_iter,idx_vars_iter); %random selection of variables

    Z = linkage(dataiter',hcvarsmethod,distfun);
        T = cluster(Z,'maxclust',k);
    
    [~,Chi2(iter),pChi2(iter)] = crosstab(T,idx_vars_res(idx_vars_iter)); 
    
    clc
end

%Permutation of variables of shuffled data
clear Chi2shuffle *pChi2shuffle;
for iter = 1:10000 %iteration
    
    %Permutation of variables
    Nvarsiter = round(.7*Nvars); %70% of original variables
    idx_vars_iter = randperm(Nvars,Nvarsiter); %random 70% indices of variables
    
    dataiter = datashuffle(idx_obs_iter,idx_vars_iter); %random selection of variables

    Z = linkage(dataiter',hcvarsmethod,distfun);
        T = cluster(Z,'maxclust',k);
    
    [~,Chi2shuffle(iter),pChi2shuffle(iter)] = crosstab(T,idx_vars_res(idx_vars_iter)); 
    
    clc
end

%Plot histogram of p-values across iterations of partitioned variables
figure('color','w')
histogram(pChi2,0:.01:1,'normalization','probability','facecolor',[1 0 0]) %real data
hold on,histogram(pChi2shuffle,0:.01:1,'normalization','probability','facecolor',[.5 .5 .5]) %shuffled data
gridxy(.05,[],'linestyle','--','color','k')
    title('Permutation of variables')
    ylabel('Probability')
    xlabel('Chi-squared test {\it p}-value')
    legend({'','data','shuffle'},'box','off')
    set(gca,'box','off')
    pause(0)

probpChi2vars = 100*mean(pChi2<0.05)

%% WITHIN VS. BETWEEN TESTS 
%% Code for Figure S1A
% Hierarchical clustering of variables
% Tests the relationship with behavioral tests

%Hierarchical clustering
k = 6; %number of clusters (behavioral tests?)

distfun = @(i,j) min([pdist2(i,j) ; pdist2(i,-j)]); %"sign-independent" Euclidean distance
hcvarsmethod='ward'; %Ward's method

Zvars = linkage(data',hcvarsmethod,distfun); %tree clustering
    Dvars = pdist(data'); %distance distributions
    sortvars = optimalleaforder(Zvars,Dvars); %indices or dendrogram-sorted variables (used throughout this script)
    Tvars = cluster(Zvars,'maxclust',k); %cluster by defined number of clusters
    COPHvars = cophenet(Zvars,Dvars) %cophenetic coefficient

%Plot dendrogram
figure('color','w'),
    [H] = dendrogram(Zvars,0,'orientation','top','reorder',sortvars,'colorthreshold',Zvars(end,3));
    ylabel('Dissimilarity')
    set(H,'linewidth',1.5,'color','k')
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    pause(0)
    
%Chi-squared test clusters x behavioral tests
[tbl,chi2,pchi2] = crosstab(Tvars,idx_vars_test); chi2, pchi2
    %Plot counts of clusters x behavioral tests
    testlabels = {'OF','FST','SIT','EPM','NOR','PPI','SPT','SB'};
    testcolor = [1 .5 .25; 0.25 .75 1; .8 0 0; .5 0 1; .5 1 .5 ; 1 .5 .5; 1 .9 0; .75 .75 .75]; %colors for each test
    figure('color','w'),bar(tbl,'stacked'); ylabel('Counts'); xlabel('Cluster')
    legend(testlabels(unique(idx_vars_test)),'box','off')
    pause(0)
    %Plot dendrogram-sorted behavioral tests' variables
    figure('color','w'),imagesc(idx_vars_test(sortvars)); colormap(testcolor); set(gca,'visible','off')
    pause(0)

%% Code for Figure S1B
% Factor analysis

nF = 7; %number of factors

[lambda,psi,~,Fstats,Fscore] = factoran(data,nF,'scores','regression','rotate','varimax'); %factor analysis

%Plot factor loadings moduli
sortvars = 1:size(data,2); %reorder indices of variables to time order
figure('color','w'),imagesc(lambda(sortvars,:)'); 
    colormap(colormaprwb); 
    caxis([-.5 .5]);  
    c=colorbar; ylabel(c,'Factor Loading'); set(c,'ytick',[-.5 .5])
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    ylabel('Factor')
    title('Factor loadings')
    pause(0)
    
%Plot factor loadings
figure('color','w'),imagesc(abs(lambda(sortvars,:)'));
    colormap(flipud(colormap('bone'))); 
    caxis([.2 .8]); 
    c=colorbar; ylabel(c,'Factor Loading (mod.)'); set(c,'ytick',[.2 .5 .8])
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    ylabel('Factor')
    title('Factor loadings')
    pause(0)
    
%% COVARIATION PATTERS
%% Code for Figure 4A-D
% Principal component analysis (PCA)

%Principal component analysis
[PC,PCscore,~,~,varexp] = pca(data);
PCscore = PC'*data'; PCscore = PCscore';

PCvarcontrib = 100*(PC.^2); %variable contributions

%PCA of shuffled data
[PCshuffle,~,~,~,varexpshuffle] = pca(datashuffle);

%Plot explained variance
figure('color','w'),
    hold on,plot(varexp,'linewidth',4,'color','k')
    hold on,plot(varexpshuffle,'linewidth',4,'color',[.75 .75 .75])
%     hold on,plot(cumsum(varexp)/sum(varexp))
%     hold on,plot(cumsum(varexpshuffle)/sum(varexpshuffle))
    ylabel('Explained variance (%)')
    xlabel('PCs')
    varexp50 = find(cumsum(varexp)/sum(varexp)>.50,1); disp([num2str(varexp50) ' PCs for 50% var. exp.'])
    varexp80 = find(cumsum(varexp)/sum(varexp)>.80,1); disp([num2str(varexp80) ' PCs for 80% var. exp.'])
    text(varexp50,varexp(varexp50)+1,'\downarrow 50%','fontsize',12)
    text(varexp80,varexp(varexp80)+1,'\downarrow 80%','fontsize',12)
    pause(0)
    
%Plot principal components' coefficients
[~,sortvars] = sort(idx_vars_res,'descend'); %order by resilience/susceptibility
figure('color','w'),plot(PC(sortvars,1:3),'linewidth',4) %plot PC1-3
    hold on,plot([1 numel(idx_vars)],[0 0],'k--')
    hold on,plot([1 numel(idx_vars)],[.2 .2],'k:')
    hold on,plot([1 numel(idx_vars)],-[.2 .2],'k:')
    gridxy(find(idx_vars_res(sortvars)==-1,1)-.5,[],'linestyle','--','color','k'); %separate resilience vs. susceptibility
    set(gca,'box','off','XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    legend({'','PC1','PC2','PC3'},'box','off')
    ylabel('PC coefficients')
%     title('PC')
    pause(0)
    
%Plot PCA feature space map (NH vs. H)
figure('color','w'),
for ik = 0:1 %groups NH(0) and H(1)
    hold on,scatter3(PCscore(find(idx_rh==ik),1),PCscore(find(idx_rh==ik),2),PCscore(find(idx_rh==ik),3),'filled') 
end
legend({'NH','H'},'box','off')
xlabel('PC1'),ylabel('PC2'),zlabel('PC3')
title('PCA')
pause(0)

%Plot average PC scores per group (NS vs. IS)
figure('color','w'),plotclusters(PCscore,idx_group,1);
    xlabel('PC')
    xlim([0.5 10])
    ylabel('PC score')
    legend({'','NS','','IS'},'box','off')
    title('PC differences')
    pause(0)
    
%Calculate Student's t-test for each PC (NS vs. IS)
clear tvaluepc pvaluetpc
for ipc = 1:size(PCscore,2)
    [~,pvaluetpc(ipc),~,tstats] = ttest2(PCscore(idx_group==0,ipc),PCscore(idx_group==1,ipc)); tvaluepc(ipc)=tstats.tstat;
end

%% Code for Figure 4E
% Linear Discriminant Model (LDM)
% Tests classification of H vs. NH by covariation patterns

clear mvclass*
for nPC = 1:varexp80
    %data to classify (PC scores)
    datatoclass = PCscore(:,1:nPC); %cumulative PCs
    %labels to classify (NH vs. H)
    idx_class = idx_rh; 

    %Get number of observations
    Nobs = size(datatoclass,1); %number of observations
    idx_train = 1:Nobs; %all observations indices
    idx_test = idx_train; %begin with indices of test equal to train
    
    % Train x test iterations
    clear Mdlclass*
    for iter = 1:1000 %iteration
        clc
        disp(['Train/test iteration: ' num2str(iter) '/1000'])
        
        %Permutation
        Ntrain = round(.7*Nobs); %70% percentage of train data
        idx_train = randperm(Nobs,Ntrain); %train (random 70%)
        idx_test = setxor(1:Nobs,idx_train); %test (remaining 30%)

        %Labels' indices
        idx_class_train = idx_class(idx_train); %correct train labels
        shuffle_class_train = shuffle(idx_class_train); %shuffled train labels

        %Linear Discriminant Model
        LDscorefun = @(x) x; %function
        Mdl = fitcdiscr(LDscorefun(datatoclass(idx_train,:)),idx_class_train);  %fit discriminant analysis classifier
        L = loss(Mdl,LDscorefun(datatoclass(idx_test,:)),idx_class(idx_test)); %loss
        mdlclass = 1-L; %classification
        mdlclass = .5+abs(mdlclass-.5); %absolute classification

        %LDM on shuffled train labels
        Mdlshuffle = fitcdiscr(LDscorefun(datatoclass(idx_train,:)),shuffle_class_train);  %fit LDM on shuffled train labels
        Lshuffle = loss(Mdlshuffle,LDscorefun(datatoclass(idx_test,:)),idx_class(idx_test)); %loss
        mdlclassshuffle = 1-Lshuffle; %classification
        mdlclassshuffle = .5+abs(mdlclassshuffle-.5); %absolute classification
        
        Mdlclass(iter) = mdlclass; %concatenate iterations
        Mdlclassshuffle(iter) = mdlclassshuffle; %concatenate iterations
    end

    
    Mdlclass = nanmean(Mdlclass);
    Mdlclassshuffle = nanmean(Mdlclassshuffle);

    mvclass(1,nPC) = Mdlclass;
    mvclassshuffle(1,nPC) = Mdlclassshuffle;
end

%Plot average classification per cumulative PC
figure('color','w'),
hold on,plot(mvclass,'linewidth',4)
hold on,plot(mvclassshuffle,'linewidth',4)
    ylim([0 1])
    legend({'data','shuffle'},'box','off','location','southeast')
    ylabel('Classification (%)')
    xlabel('Principal Components')
    pause(0)

%% Code for Figure S2
% Permutation tests of PCA
% Tests the emergence of similar covariation patterns in partitioned data

%Whole-data PC coeff. and scores
[PC,PCscore] = pca(data);
PCscore = PC'*data'; PCscore = PCscore';

%Get numbers of observations and relevant PCs
Nobs = size(data,1);
Npc = varexp80;

clear PCiter* PCscoreiter* *Cpc*
for iter = 1:10000 %iteration
    
    Niter = round(.7*Nobs); %70% of observations
    idx_iter = randperm(Nobs,Niter); %random
    
    dataiter = data(idx_iter,:); %randomly select 70%
    datashuffleiter = datashuffle(idx_iter,:); %randomly select 70% of shuffled data
    
    [PCiter(:,:,iter),PCscoreiter(:,:,iter)] = pca(dataiter); %PCA of partitioned (70%) iteration data
    [PCitershuffle(:,:,iter),PCscoreitershuffle(:,:,iter)] = pca(datashuffleiter); %PCA of partitioned (70%) iteration shuffled data
    
    for npc = 1:Npc
        %Correlation between whole-data PCs x iteration PCs
        [cpc_coeff] = corr(PC(:,npc),PCiter(:,:,iter)); %coeffs
        [cpc_score] = corr(PCscore(idx_iter,npc),PCscoreiter(:,:,iter)); %scores
        
        [cpc_coeff_shuffle] = corr(PC(:,npc),PCitershuffle(:,:,iter)); 
        [cpc_score_shuffle] = corr(PCscore(idx_iter,npc),PCscoreitershuffle(:,:,iter)); 
        
        %Concatenate magnitude of PC x PC correlations
        Cpc_coeff(npc,:,iter) = abs(cpc_coeff); 
        Cpc_score(npc,:,iter) = abs(cpc_score); 
        
        Cpc_coeff_shuffle(npc,:,iter) = abs(cpc_coeff_shuffle); 
        Cpc_score_shuffle(npc,:,iter) = abs(cpc_score_shuffle); 
        
        %Get greatest correlation between whole-data PCs x iteration PCs
        MaxCpc_coeff(npc,iter) = max(abs(cpc_coeff)); 
        MaxCpc_score(npc,iter) = max(abs(cpc_score)); 
        MaxCpc_coeff_shuffle(npc,iter) = max(abs(cpc_coeff_shuffle)); 
        MaxCpc_score_shuffle(npc,iter) = max(abs(cpc_score_shuffle)); 
    end
end

%Plot correlation matrix of median whole-data PC x iterations PC
figure('color','w')
    imagesc(nanmedian(Cpc_coeff,3)); %data
    colormap(flipud(colormap('bone'))); 
    caxis([.2 .8])
    c=colorbar; ylabel(c,'Median corr. ({\itr})')
    xlim([.5 varexp80+.5]); 
    ylabel('Whole data PC')
    xlabel('Iteration data PC')
    axis square
    pause(0)
    
figure('color','w')
    imagesc(nanmedian(Cpc_coeff_shuffle,3)); %shuffle
    colormap(flipud(colormap('bone'))); 
    caxis([.2 .8])
    c=colorbar; ylabel(c,'Median corr.({\itr})')
    xlim([.5 varexp80+.5]); 
    ylabel('Whole data PC')
    xlabel('Iteration data PC')
    axis square
    pause(0)

%Plot median of maximum iteration PC correlation
figure('color','w')
    hold on,plot(nanmedian(MaxCpc_coeff,2),'linewidth',4)
    hold on,plot(nanmedian(MaxCpc_score,2),'linewidth',4)
    hold on,plot(nanmedian(MaxCpc_coeff_shuffle,2),'linewidth',4)
    hold on,plot(nanmedian(MaxCpc_score_shuffle,2),'linewidth',4)
    legend({'coeff.','scores','shuffle coeff.','shuffle scores'},'box','off')
    xlim([.5 varexp80+.5]); 
    ylabel('Median of maximum iteration PCs corr. ({\itr})')
    xlabel('Whole data PC')
    pause(0)

%% INDIVIDUAL CLUSTERING
%For individual clustering analysis, we used the 6 top relevant variables
idx_vars = idx_vars_top;    
    data = DATA(:,idx_vars);
    varlabels = VARLABELS(idx_vars);
idx_vars_res = idx_vars_res_top;
idx_vars_test = idx_vars_test_top;

%Normalize
data = normalize(data,1,'zscore');

% %% Shuffled data
datashuffle = shuffle(data,1); %shuffle obs
for idx_rat = 1:size(datashuffle,1)
    datashuffle(idx_rat,:) = shuffle(datashuffle(idx_rat,:),2); %shuffle vars
end

%% Code for Figure 5A-B
% Semi-supervised hierarchical clustering of individuals

datatoclust = data;

clustmethod = 'linkage'; %hierarchical clustering
distmetric = 'euclidean';

klist = [2:10]; %number of clusters to test

% %% Evaluate numbers of clusters
%evaluate number of clusters (silhouette values)
evalK = evalclusters(datatoclust,clustmethod,'silhouette','distance',distmetric,'klist',klist);  

%Plot silhouette values per number of clusters
figure('color','w'),plot(evalK); 
    text(evalK.OptimalK,evalK.CriterionValues(find(klist==evalK.OptimalK))+.005,'\downarrow','fontsize',20)
    title('Hierarchical clustering')
    pause(0)

% %% Chi-squared test number of clusters x NS vs. IS
clear Chi2 pChi2
for k = klist
    
    Zobs = linkage(datatoclust,'ward',distmetric);
    idx_cluster = cluster(Zobs,'maxclust',k);
    
    %Chi-squared test clusters x NS vs. IS
    [~,chi2,pchi2] = crosstab(idx_group,idx_cluster); 
   
    Chi2(k) = chi2;
    pChi2(k) = pchi2;
end

%Plot Chi-squared test results per number of clusters
figure('color','w'),
    hold on,plot(pChi2);
    hold on,plot([1 k],[0.05 .05], '--k'); 
    xlim([2 k]) 
    ylim([0 .2])
    ylabel('Chi-squared test {\itp}-value')
    xlabel('Number of Clusters')
    title('Hierarchical clustering')
    pause(0)
    
%% Code for Figure 5C-F
% Hierarchical clustering of individuals

datatoclust = data;

k = 7; %number of clusters

%Hierarchical clustering of observations
[~,sortvars] = sort(idx_vars_res,'descend'); %order variables by susceptibility/resilience
distmetric = 'euclidean';
Zobs = linkage(datatoclust,'ward',distmetric);
    Dobs = pdist(datatoclust);
    sortobs = optimalleaforder(Zobs,Dobs);
    COPHobs = cophenet(Zobs,Dobs)
    
idx_cluster = cluster(Zobs,'maxclust',k); %clustering

%Plot dendrogram
figure('color','w'),
    [h]=dendrogram(Zobs,0,'orientation','left','reorder',sortobs); set(h,'linewidth',1.5,'color','k')
    xlabel('Dissimilarity')
    pause(0)

%Plot data matrix 
figure('color','w'),
    imagesc(data(sortobs,sortvars)); 
    axis xy, 
    caxis([-2 2]); 
    colormap(colormaprwb); 
    c=colorbar; ylabel(c,'Z-score') 
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
    pause(0)
    
%Plot dendrogram-sorted groups
figure('color','w'),
    imagesc(idx_group(sortobs)); 
    colormap([1 0 0; 0 0 1]); 
    set(gca,'visible','off')
    pause(0)

%sort clusters' ID by SB escape performance
idx_cluster = sortk(idx_cluster,data(:,find(contains(varlabels,'SB (Escape Failures)'))));  

%Chi-squared test clusters x NS vs. IS
[tbl,chi2,pchi2] = crosstab(idx_group,idx_cluster); chi2, pchi2
    tblcperg = 100*tbl./repmat(sum(tbl,2),1,size(tbl,2)) %table of clusters per group
    tblgperc = 100*tbl./repmat(sum(tbl,1),size(tbl,1),1) %table of groups per cluster
    idxsortclust = 1: size(tbl,2); %do not sort

%Plot counts of groups per cluster
figure('color','w'),
    b=bar(tbl(:,idxsortclust)',1); set(b,'linewidth',1)
    title('Hierarchical clustering')
    l = legend({'NS','IS'},'box','off');
    ylabel('Counts')
    xlabel('Clusters')
    pause(0)
    
%Plot clusters' profiles
% clustercolors = colormap('lines'); %lines colormap
clustercolors = (colormap(colormaprb)); 
    idx_cmap = round(linspace(1,size(clustercolors,1),k)); 
    clustercolors = clustercolors(idx_cmap,:); %colormap

clear centroids
for ik = 1:k
    figure('color','w'),
    hold on,boundedlinepadrao([],data(find(idx_cluster==idxsortclust(ik)),sortvars),clustercolors(ik,:),1,1); title(['Cluster ' num2str(ik)])

    hold on,plot([1 numel(idx_vars)],[0 0],'k-')
    hold on,plot([1 numel(idx_vars)],[.5 .5],'k--')
    hold on,plot([1 numel(idx_vars)],-[.5 .5],'k--')

    ylabel('Z-score')
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
end

%% Code for Figure S4A-B
% Semi-supervised k-means clustering of individuals

datatoclust = data;

clustmethod = @(x,k) kmeans(x,k,'replicates',10000,'maxiter',100); %k-means with many replicates (robust)
distmetric = 'sqeuclidean';

klist = [2:10]; %number of clusters to test

% %% Evaluate numbers of clusters
%evaluate number of clusters (silhouette values)
evalK = evalclusters(datatoclust,clustmethod,'silhouette','distance',distmetric,'klist',klist);  

%Plot silhouette values per number of clusters
figure('color','w'),plot(evalK); 
    text(evalK.OptimalK,evalK.CriterionValues(find(klist==evalK.OptimalK))+.005,'\downarrow','fontsize',20)
    title('K-means')
    pause(0)

% %% Chi-squared test number of clusters x NS vs. IS
clear Chi2 pChi2
for k = klist
    
    [idx_cluster,centroidsK,sumD] = kmeans(datatoclust,k,'distance',distmetric,...
        'Replicates',10000,'maxiter',100,'Display','final');
    
    %Chi-squared test clusters x NS vs. IS
    [~,chi2,pchi2] = crosstab(idx_group,idx_cluster); 
   
    Chi2(k) = chi2;
    pChi2(k) = pchi2;
end

%Plot Chi-squared test results per number of clusters
figure('color','w'),
    hold on,plot(pChi2);
    hold on,plot([1 k],[0.05 .05], '--k'); 
    xlim([2 k]) 
    ylim([0 1])
    ylabel('Chi-squared test {\itp}-value')
    xlabel('Number of Clusters')
    title('K-means')
    pause(0)

%% Code for Figure S4D-E
% K-means clustering of individuals

datatoclust = data;

k = 6; %number of clusters

%K-means clustering
[~,sortvars] = sort(idx_vars_res,'descend'); %order variables by susceptibility/resilience
distmetric = 'sqeuclidean';
[idx_cluster,centroidsK,sumD] = kmeans(datatoclust,k,'distance',distmetric,...
    'Replicates',10000,'maxiter',100,'Display','final');

%sort clusters' ID by SB escape performance
idx_cluster = sortk(idx_cluster,data(:,find(contains(varlabels,'SB (Escape Failures)'))));  

%Chi-squared test clusters x NS vs. IS
[tbl,chi2,pchi2] = crosstab(idx_group,idx_cluster); chi2, pchi2
    tblcperg = 100*tbl./repmat(sum(tbl,2),1,size(tbl,2)) %table of clusters per group
    tblgperc = 100*tbl./repmat(sum(tbl,1),size(tbl,1),1) %table of groups per cluster
    idxsortclust = 1: size(tbl,2); %do not sort

%Plot counts of groups per cluster
figure('color','w'),
    b=bar(tbl(:,idxsortclust)',1); set(b,'linewidth',1)
    title('Hierarchical clustering')
    l = legend({'NS','IS'},'box','off');
    ylabel('Counts')
    xlabel('Clusters')
    pause(0)
    
%Plot clusters' profiles
clustercolors = (colormap(colormaprb)); 
    idx_cmap = round(linspace(1,size(clustercolors,1),k)); 
    clustercolors = clustercolors(idx_cmap,:); %colormap

clear centroids
for ik = 1:k
    figure('color','w'),
    hold on,boundedlinepadrao([],data(find(idx_cluster==idxsortclust(ik)),sortvars),clustercolors(ik,:),1,1); title(['Cluster ' num2str(ik)])

    hold on,plot([1 numel(idx_vars)],[0 0],'k-')
    hold on,plot([1 numel(idx_vars)],[.5 .5],'k--')
    hold on,plot([1 numel(idx_vars)],-[.5 .5],'k--')

    ylabel('Z-score')
    set(gca,'XTick',[1:length(idx_vars)],'XTickLabel',varlabels(sortvars),'XTickLabelRotation',90) 
end

%Plot silhouette values
figure('color','w'),
    [silh,h] = silhouette(datatoclust,idx_cluster,distmetric);
    xlabel('Silhouette Values')
    ylabel('Cluster')
    title('Silhouette')
    pause(0)
