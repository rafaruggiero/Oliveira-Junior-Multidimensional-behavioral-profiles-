
[ndata, tdata, alldata] = xlsread('Data_Table.xlsx');
textdata=tdata(:,2:end);

[idx_row, idx_col]=find(isnan(ndata));   % Finding NaN values indexes

NANinfo = (alldata(1, idx_col+1));
NANinfo(2,:) = (alldata(idx_row+1, 2))';
NANinfo(3,:) = (alldata(idx_row+1, 1))';           

% Replacing NaN values by column median for each experimental group

numdata=ndata;

for i = 1:length(idx_row)                        
    
    if numdata(idx_row(i),1)==1
        numdata(idx_row(i),idx_col(i)) = median(ndata(ndata(:,1)==1,idx_col(i)),'omitnan');
   
    else
        numdata(idx_row(i),idx_col(i)) = median(ndata(ndata(:,1)==0,idx_col(i)),'omitnan');
      
    end
end

clearvars -except alldata numdata textdata

%% Multivariate Outliers

IS = find(numdata(:,1)==1);
NS = find(numdata(:,1)==0);
g = cell({IS,NS});

datavars = zscore(numdata(:,[6 31 51 54 71 75 76 78 97 98])); % The 10 principal behavioral measures

for i=1:2
    DataSet = DDOutlier.dataSet(datavars(g{i},:),'euclidean');
    [~,max_nb] = DDOutlier.NaNSearching(DataSet);
    [nofs] = DDOutlier.NOFs(DataSet,max_nb);
    [lofs] = DDOutlier.LOFs(DataSet,max_nb);
    figure, scatter(lofs,1:length(lofs),'filled','k')
    t={'IS','NS'};
    title(t{i}), ylabel('Observations'), xlabel('LOF')
    posit{1,i} = find(lofs>1.1);
end
clear lofs nofs max_nb DataSet t datavars 

posit{1,2} = posit{1,2}+length(IS);
disp(['Outlier position in IS group: ' num2str(posit{1,1}) '. Outlier position in NS group: ' num2str(posit{1,2}) '.'])


numdata_out=numdata; % All data (101/101 variables)
numdata_out([posit{1,1} posit{1,2}],:)=[]; % Removes multivariate outliers -> LOF>1.1
%znumdata = zscore(numdata_out(:,2:end),1);

clearvars -except alldata numdata textdata numdata_out

%% Variable Selection and Data standardization

% Relevant data (33/101 variables) % The 33 principal behavioral measures removing multivariate outlier
relevant_data=numdata_out(:,[1 6 8 9 14 18 22 24 31 37 43 45 51 52 54 55 56 57 58 62 71 75 76 77 78 79 94 95 96 97 98 99 100 101]);
relevant_data_label = textdata(1,[1 6 8 9 14 18 22 24 31 37 43 45 51 52 54 55 56 57 58 62 71 75 76 77 78 79 94 95 96 97 98 99 100 101]);
relevant_data(:,2:end) = zscore(relevant_data(:,2:end),1);

% Principal data (6/101 variables)
prindata = numdata_out(:,[1 6 31 51 54 97 98]); % The 6 principal behavioral measures removing multivariate outlier
prindata_noZ = prindata; % Sem zscore
prindata(:,2:end) = zscore(prindata(:,2:end),1);
prindata_label = textdata(1,[1 6 31 51 54 97 98]);

%% Clustering Resistant vs Helpless

C_data = numdata_out(:,[1 98 101]); % Selecting Variables: Latency to Escape and Escape Failures
C_data(:,2:end) = zscore(C_data(:,2:end)); 

evalK = evalclusters(C_data(:,2:end),'kmeans','silhouette','klist',1:4), figure, plot(evalK)

rng('default') % For reproducibility
[idx_kcluster,Kcentroids,sumD] = kmeans(C_data(:,2:end),2,'Replicates',100,'Display','final');

% idx_kcluster -> 2==Helpless, 1==Resistant

% Manually changing classes for easier identification
 idx_kcluster(find(idx_kcluster==2))=20;
 idx_kcluster(find(idx_kcluster==1))=2;
 idx_kcluster(find(idx_kcluster==20))=1;
% 1==Helpless, 2==Resistant

silhouette(C_data(:,2:end),idx_kcluster)
set(gcf,'color','white'), set(gca, 'YTickLabel',{'H','R'}, 'fontname','helvetica','fontsize',12,...
    'linewidth',1.5,'xcolor','k','ycolor','k','box','off')

% Latency to escape vs. escape failures
figure, scatter(numdata_out(numdata_out(:,1)==1,95),numdata_out(numdata_out(:,1)==1,98), 'filled', 'r');  % IS
hold on, scatter(numdata_out(numdata_out(:,1)==0,95),numdata_out(numdata_out(:,1)==0,98), 'filled', 'b'); % NS
legend('IS', 'NS', 'location', 'southeast', 'box', 'off')
xlabel('Escape failures'), ylabel('Mean latency to escape (sec)'), set(gcf,'color','white')
set(gca,'ytick',2:2:10,'xtick',0:10:30,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')

figure, scatter(C_data(idx_kcluster(:,1)==1,2),C_data(idx_kcluster(:,1)==1,3), 'filled', 'k');  % Helpless
hold on, scatter(C_data(idx_kcluster(:,1)==2,2),C_data(idx_kcluster(:,1)==2,3), 'filled', 'g'); % Resistant
legend('H', 'R', 'location', 'southeast', 'box', 'off')
xlabel('Escape failures (Z-score)'), ylabel('Mean latency to escape (Z-score)'), set(gcf,'color','white')
set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
% print -dpdf -painters Sillhuette


% X²
[tbl,chi2,pchi2] = crosstab(prindata(:,1),idx_kcluster); pchi2

figure,b=bar(tbl(:,1:2)',1); set(b,'linewidth',1.5)
b(1).FaceColor = [0 0 1];
b(2).FaceColor = [1 0 0];
l1 = legend({'NS','IS'},'box','off');
ylabel('Individuals'), xlabel('Cluster'), set(gcf,'color','white')
%print -dpdf -painters clustX2


clearvars -except alldata numdata textdata numdata_out relevant_data relevant_data_label prindata prindata_noZ prindata_label idx_kcluster

%% VARIABLES CORRELATION

% Hierarchical clustering based on correlation
Zvars = linkage(relevant_data(:,2:end)','ward','correlation');
Dvars = pdist(relevant_data(:,2:end)');
sortvars = optimalleaforder(Zvars,Dvars); 
Tvars = cluster(Zvars,'maxclust',2);
    figure,[H] = dendrogram(Zvars,0,'orientation','top','reorder',sortvars,'colorthreshold',Zvars(end,3));
    set(H,'linewidth',1.5), set(gcf, 'color','white'), ylabel('Dissimilarity')
    set(gca,'XTick',[1:size(relevant_data(:,2:end),2)],'XTickLabel',relevant_data_label(sortvars+1),'XTickLabelRotation',90) 
% COPHvars = cophenet(Zvars,Dvars)
% Ivars = inconsistent(Zvars);


% Matrix correlation
[c_reldata, p_reldata] = corr(relevant_data(:,2:end));
c_reldata(c_reldata==1)=0; % Diagonal whitening
figure, imagesc (c_reldata(sortvars,sortvars)), colormap redblue, c=colorbar; caxis([-.5 .5]);
set(c,'fontsize',12,'color','k','linewidth',1.5,'ticks',[-.5 0 .5])
ylabel(c,'Correlation'), set(gcf, 'position', [050 35 790 650], 'Color', 'white')
set(gca,'XTick',1:size(c_reldata,1),'YTick',1:size(c_reldata,2), 'XTickLabel', relevant_data_label(sortvars+1), 'YTickLabel', relevant_data_label(sortvars+1),...
    'XTickLabelRotation', 90, 'fontname','helvetica', 'fontsize',7,'linewidth',1.5,'xcolor','k','ycolor','k','box','on')
 %print -dpdf -painters Corr_allvars


 % Test for multicollinearity
v=vif(prindata(:,2:end)); 
bar(v,0.5,'k'), set(gcf, 'color','white'), ylabel('VIF Values')
set(gca,'Ydir','reverse','XAxisLocation','top','XTickLabel',[],'TickDir','out', 'YLim', [0 1.5], 'YTick', 0:0.5:1.5)
set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
% print -dpdf -painters Vif_values

clearvars -except alldata numdata textdata numdata_out relevant_data relevant_data_label prindata prindata_noZ prindata_label idx_kcluster

%% Principal Component Analysis (PCA)

 x = prindata; varslabel = prindata_label % 6/101 variables

[PCcoef, PCscore, PCvar, ~, PCexpvar, Emean] = pca(x(:,2:end)); % pca(prindata(find(idx_kcluster==1),:))

% Explained Variance
figure, plot(PCexpvar(1:size(PCexpvar,1)),'k','linewidth',2);
set(gca,'box','off'), xlabel('Principal Components (PCs)', 'FontWeight','bold'), ylabel('Explained Variance (%)','FontWeight','bold'); 
set(gcf,'color','white'), set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')

if size(PCexpvar,2)>6
    hold on, plot([PCexpvar(1),sum(PCexpvar(1:2)),sum(PCexpvar(1:3)),sum(PCexpvar(1:4)), sum(PCexpvar(1:5)),...
        sum(PCexpvar(1:6)), sum(PCexpvar(1:7)), sum(PCexpvar(1:8)), sum(PCexpvar(1:9)), sum(PCexpvar(1:10))],'r');
else
    hold on, plot([PCexpvar(1),sum(PCexpvar(1:2)),sum(PCexpvar(1:3)),sum(PCexpvar(1:4)), sum(PCexpvar(1:5)),...
        sum(PCexpvar(1:6))],'r');
end
% xlim([0 10])

% PC1 vs. PC2
    % IS vs. NS
    figure, scatter(PCscore(x(:,1)==1,1),PCscore(x(:,1)==1,2), 'filled', 'red');  % 2D
    hold on, scatter(PCscore(x(:,1)==0,1),PCscore(x(:,1)==0,2), 'filled', 'blue');
    xlabel('PC1', 'FontWeight','bold'), ylabel('PC2','FontWeight','bold');
    line([0 0],[ylim],'color','black','linestyle','--','linewidth',1)
    line([xlim],[0 0],'color','black','linestyle','--','linewidth',1)
    m1 = mean(PCscore(x(:,1)==1,1:2));
    m2 = mean(PCscore(x(:,1)==0,1:2));
    plot(m1(1), m1(2),'Marker','+','color','red','MarkerSize',8,'linewidth',1.5)
    plot(m2(1), m2(2),'Marker','+','color','blue','MarkerSize',8,'linewidth',1.5)
    legend({'IS', 'NS'}, 'location', 'northwest'), grid;
    set(gcf,'color','white'), set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    set (gcf, 'position', [300 65 700 600]);
    %print -dpdf -painters PC1vsPC2_ISNS
    
    % Hel vs. Res
    figure, scatter(PCscore(idx_kcluster==1,1),PCscore(idx_kcluster==1,2), 'filled','k');  % 2D
    hold on, scatter(PCscore(idx_kcluster==2,1),PCscore(idx_kcluster==2,2), 'MarkerFaceColor', [0.2 0.8 0.2], 'MarkerEdgeColor', [0.2 0.8 0.2]);
    xlabel('PC1', 'FontWeight','bold'), ylabel('PC2','FontWeight','bold');
    line([0 0],[ylim],'color','black','linestyle','--','linewidth',1)
    line([xlim],[0 0],'color','black','linestyle','--','linewidth',1)
    m1 = mean(PCscore(idx_kcluster==1,1:2));
    m2 = mean(PCscore(idx_kcluster==2,1:2));
    plot(m1(1), m1(2),'Marker','+','color','k','MarkerSize',8,'linewidth',1.5)
    plot(m2(1), m2(2),'Marker','+','MarkerFaceColor', [0.2 0.8 0.2], 'MarkerEdgeColor', [0.2 0.8 0.2],'MarkerSize',8,'linewidth',1.5)
    legend({'H', 'R'}, 'location', 'northwest'), grid;
    set(gcf,'color','white'), set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    set (gcf, 'position', [300 65 700 600]);
    %print -dpdf -painters PC1vsPC2_HelRes

% PC3D

    % IS vs. NS
    figure, scatter3(PCscore(x(:,1)==1,1),PCscore(x(:,1)==1,2), PCscore(x(:,1)==1,3), 'filled', 'red')
    hold on, scatter3(PCscore(x(:,1)==0,1),PCscore(x(:,1)==0,2), PCscore(x(:,1)==0,3), 'filled', 'blue')
    xlabel('PC1', 'FontWeight','bold'), ylabel('PC2','FontWeight','bold'), zlabel('PC3','FontWeight','bold')
    line([0 0],[ylim],[0 0],'color','black','linestyle','--','linewidth',1)
    line([xlim],[0 0], [0 0],'color','black','linestyle','--','linewidth',1)
    line([0 0],[0 0], [zlim],'color','black','linestyle','--','linewidth',1)
    m1 = mean(PCscore(x(:,1)==1,1:2));
    m2 = mean(PCscore(x(:,1)==0,1:2));
    plot(m1(1), m1(2),'Marker','+','color','red','MarkerSize',8,'linewidth',1.5)
    plot(m2(1), m2(2),'Marker','+','color','blue','MarkerSize',8,'linewidth',1.5)
    legend({'IS', 'NS'}, 'location', 'northwest'), %grid;
    set(gcf,'color','white'), set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    set (gcf, 'position', [300 65 700 600]);
    %print -dpdf -painters PC3_ISNS

    % Hel vs. Res
    figure, scatter3(PCscore(idx_kcluster==1,1),PCscore(idx_kcluster==1,2),PCscore(idx_kcluster==1,3), 'filled','k');
    hold on, scatter3(PCscore(idx_kcluster==2,1),PCscore(idx_kcluster==2,2),PCscore(idx_kcluster==2,3),...
        'MarkerFaceColor', [0.2 0.8 0.2], 'MarkerEdgeColor', [0.2 0.8 0.2]);
    xlabel('PC1', 'FontWeight','bold'), ylabel('PC2','FontWeight','bold'), zlabel('PC3','FontWeight','bold')
    line([0 0],[ylim],[0 0],'color','black','linestyle','--','linewidth',1)
    line([xlim],[0 0], [0 0],'color','black','linestyle','--','linewidth',1)
    line([0 0],[0 0], [zlim],'color','black','linestyle','--','linewidth',1)
    m1 = mean(PCscore(idx_kcluster==1,1:2));
    m2 = mean(PCscore(idx_kcluster==2,1:2));
    plot(m1(1), m1(2),'Marker','+','color','k','MarkerSize',8,'linewidth',1.5)
    plot(m2(1), m2(2),'Marker','+','MarkerFaceColor', [0.2 0.8 0.2], 'MarkerEdgeColor', [0.2 0.8 0.2],'MarkerSize',8,'linewidth',1.5)
    legend({'H', 'R'}, 'location', 'northwest'), %grid;
    set(gcf,'color','white'), set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    set (gcf, 'position', [300 65 700 600]);
    % print -dpdf -painters PC3_HelRes
    
% PC Coef
    figure, subplot(1,1,1),plot(PCcoef(:,1:3),'linewidth',4)
    hold on,plot([1 size(x,2)-1],[0 0],'k-')
    hold on,plot([1 size(x,2)-1],[.4 .4],'k--')
    hold on,plot([1 size(x,2)-1],-[.4 .4],'k--')
    set(gca,'YLim',[-0.8 0.8],'YTick',-0.8:0.2:0.8,'XTick',[1:size(x,2)],...
        'XTickLabel',varslabel(2:end),'XTickLabelRotation',90, 'box', 'off')
    set(gcf,'color','white'), set(gca,'fontname','helvetica','fontsize',10,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    title({'\color{blue}PC1', '\color{orange}PC2', '\color{yellow}PC3'}), ylabel('PC Coefficient')
    set (gcf, 'position', [300 45 570 630])
    %print -dpdf -painters PC_Coef
 

% PCs vs. individuals (IS x NS)
    ind = [PCscore(x(:,1)==1,1); PCscore(x(:,1)==0,1)];
    g1 = repmat({'IS'},length(find(x(:,1)==1)),1);
    g2 = repmat({'NS'},length(find(x(:,1)==0)),1);
    g = [g1; g2];
    
    figure, 
    subplot(1,3,1), boxplot(ind,g), title('PC1','Fontsize',12), set(gcf,'color','white')
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    %hold on, line([1:2],[2.8 2.8],'color','k'), plot(1.5,3,'Marker','*','color','black','MarkerSize',8)
    
    clear ind, ind = [PCscore(x(:,1)==1,2); PCscore(x(:,1)==0,2)];
    
    subplot(1,3,2), boxplot(ind,g), title('PC2','Fontsize',12), set(gcf,'color','white')
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    %hold on, line([1:2],[2.8 2.8],'color','k'), plot(1.5,3,'Marker','*','color','black','MarkerSize',8)
    
    clear ind, ind = [PCscore(x(:,1)==1,3); PCscore(x(:,1)==0,3)];
    
    subplot(1,3,3), boxplot(ind,g), title('PC3','Fontsize',12), set(gcf,'color','white')
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    %hold on, line([1:2],[2.8 2.8],'color','k'), plot(1.5,3,'Marker','*','color','black','MarkerSize',8)
    
    %subplot(1,3,1), ylim([-3,3])
    %print -dpdf -painters PCA_ISvsNS

    % Stats (IS x NS)
    for i=1:3 % PCs
    [h,~] = kstest2(PCscore(x(:,1)==1,i),PCscore(x(:,1)==0,i));
        if h==0
            [~,p,~,tstats] = ttest2(PCscore(x(:,1)==1,i),PCscore(x(:,1)==0,i));
            disp(['PC' num2str(i) ': p=' num2str(p) ', tstat: ' num2str(tstats.tstat) ', df: ' num2str(tstats.df)]);
        else
            [p,~,wstats] = ranksum(PCscore(x(:,1)==1,i),PCscore(x(:,1)==0,i));
            disp(['PC' num2str(i) ': p=' num2str(p) ', zval: ' num2str(wstats.zval) ', ranksum: ' num2str(wstats.ranksum)]);
        end
    clear h p tstats wstats
    end


% PCs vs. individuals (Hel x Res)
    clear ind g1 g2 g
    ind = [PCscore(idx_kcluster==1,1); PCscore(idx_kcluster==2,1)];
    g1 = repmat({'Hel'},length(find(idx_kcluster==1)),1);
    g2 = repmat({'Res'},length(find(idx_kcluster==2)),1);
    g = [g1; g2];
    
    figure, 
    subplot(1,3,1), boxplot(ind,g), title('PC1','Fontsize',12), set(gcf,'color','white')
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    %hold on, line([1:2],[2.8 2.8],'color','k'), plot(1.5,3,'Marker','*','color','black','MarkerSize',8)
    
    clear ind, ind=[PCscore(idx_kcluster==1,2); PCscore(idx_kcluster==2,2)]
    
    subplot(1,3,2), boxplot(ind,g), title('PC2','Fontsize',12), set(gcf,'color','white')
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    %hold on, line([1:2],[2.8 2.8],'color','k'),
    %plot(1.5,3,'Marker','*','color','black','MarkerSize',8), plot(1.5,3,'Marker','*','color','black','MarkerSize',8)

    clear ind, ind=[PCscore(idx_kcluster==1,3); PCscore(idx_kcluster==2,3)]
    
    subplot(1,3,3), boxplot(ind,g), title('PC3','Fontsize',12), set(gcf,'color','white')
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
    %hold on, line([1:2],[2.8 2.8],'color','k'),
    %plot(1.5,3,'Marker','*','color','black','MarkerSize',8), plot(1.5,3,'Marker','*','color','black','MarkerSize',8)

    clear ind, ind=[PCscore(idx_kcluster==1,4); PCscore(idx_kcluster==2,4)]
   
    %subplot(1,3,1), ylim([-3,3])
    %print -dpdf -painters PCA_HelRes

    % Stats (Hel x Res)
    for i=1:3
    [h,~] = kstest2(PCscore(idx_kcluster==1,i),PCscore(idx_kcluster==2,i));
        if h==0
            [~,p,~,tstats] = ttest2(PCscore(idx_kcluster==1,i),PCscore(idx_kcluster==2,i));
            disp(['PC' num2str(i) ': p=' num2str(p) ', tstat: ' num2str(tstats.tstat) ', df: ' num2str(tstats.df)]);
        else
            [p,~,wstats] = ranksum(PCscore(idx_kcluster==1,i),PCscore(idx_kcluster==2,i));
            disp(['PC' num2str(i) ': p=' num2str(p) ', zval: ' num2str(wstats.zval) ', ranksum: ' num2str(wstats.ranksum)]);
        end
    clear h p tstats wstats
    end

clearvars -except alldata numdata textdata numdata_out relevant_data relevant_data_label prindata prindata_noZ prindata_label idx_kcluster

%% Clustering Multivariated Phenotypes 

for i=1:1000
    evalK = evalclusters(prindata(:,2:end),'kmeans','silhouette','klist',2:10); %figure, plot(evalK)
    nK(i) = evalK.OptimalK;
end
histogram(nK), xlabel('Optimal number of clusters'), ylabel('Iterations (1000)')
set(gcf,'color','white'), set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
%print -dpdf -painters kmeans_iterations1000

rng(2) % For reproducibility
k=6
idx_multicluster = kmeans(prindata(:,2:end),k,'replicates',1000,'Display','final');

silhouette(prindata(:,2:end),idx_multicluster)
[s,h]=silhouette(prindata(:,2:end),idx_multicluster);
s_value=mean(s)

% Cluster Visualization
%idxclust = 1:k;
idxclust = [6 1 4 5 2 3] % Choosing cluster position
altposition=idx_multicluster;
altposition(find(altposition==6))=10; altposition(find(altposition==1))=20; altposition(find(altposition==4))=30;
altposition(find(altposition==5))=40; altposition(find(altposition==2))=50; altposition(find(altposition==3))=60;
altposition(find(altposition==10))=1; altposition(find(altposition==20))=2; altposition(find(altposition==30))=3;
altposition(find(altposition==40))=4; altposition(find(altposition==50))=5; altposition(find(altposition==60))=6;
silhouette(prindata(:,2:end),altposition)

figure
for i=1:k
   cmap = colormap('lines');
   subplot(k+2,1,i)
   %plot(mean(prindata(idx_multicluster==idxclust(i),:),1));
   hold on,boundedlinepadrao([],prindata(idx_multicluster==idxclust(i),2:end),cmap(i,:),1,1); title(['Cluster ' num2str(i)])
   hold on,plot([1 size(prindata(:,2:end),2)],[0 0],'k-')
   hold on,plot([1 size(prindata(:,2:end),2)],[1 1],'k--')
   hold on,plot([1 size(prindata(:,2:end),2)],-[1 1],'k--')
   ylim([-2 2]), ylabel('Z-score')
   set(gca,'XTick',[],'XTickLabel',[]), set(gcf,'color','white')
   set(gca,'fontname','helvetica','fontsize',10,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
end
set(gca,'XTick',[1:size(prindata(:,2:end),2)],'XTickLabel',prindata_label(:,2:end),'XTickLabelRotation',90, 'fontsize',10)
set(gcf, 'position', [450 45 450 640])
%print -dpdf -painters Silhouette_multivar

% CHI²
[tbl,chi2,pchi2] = crosstab(prindata(:,1),idx_multicluster); pchi2

figure,b=bar(tbl(:,idxclust)',1); set(b,'linewidth',1.5)
b(1).FaceColor = [0 0 1];
b(2).FaceColor = [1 0 0];
legend({'NS','IS'},'box','off');
ylabel('Individuals'), xlabel('Cluster'), set(gcf,'color','white')
set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
%print -dpdf -painters x2_ISNS_kmeans

[tbl,chi2,pchi2] = crosstab(idx_kcluster,idx_multicluster); pchi2

figure,b=bar(tbl(:,idxclust)',1); set(b,'linewidth',1.5)
b(1).FaceColor = [0 0 0];
b(2).FaceColor = [0.2 0.8 0.2];
l1 = legend({'Hel','Res'},'box','off');
ylabel('Individuals'), xlabel('Cluster'), set(gcf,'color','white')
set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')
%print -dpdf -painters x2_HelRes_kmeans

% Clusters mean and confidence intervals
for i=1:k
    x = prindata_noZ(idx_multicluster==i,2:end);
    M(i,:) = mean(x,1);
    SEM(i,:) = std(x)/sqrt(length(x));
end

% Clusters Z-scores and SEM
for i=1:k
    clustermean(i,:) = mean(prindata(altposition==i,2:7),1) 
    clusterSEM(i,:) = std(prindata(altposition==i,2:7))/sqrt(size(prindata(altposition==i,2:7),1)) % SEM
end

clearvars -except alldata numdata textdata numdata_out relevant_data relevant_data_label prindata prindata_label idx_kcluster

%% Univariate Analysis - violin plots

% IS vs. NS
g1 = repmat({'IS'},length(find(numdata_out(:,1)==1)),1);
g2 = repmat({'NS'},length(find(numdata_out(:,1)==0)),1);
groups_ISNS = [g1; g2];


% H vs. R
groups_HR = cell(length(idx_kcluster),1);
gg1 = find(idx_kcluster(:,1)==1);
gg2 = find(idx_kcluster(:,1)==2);
groups_HR(gg1) = repmat({'H'},length(gg1),1);
groups_HR(gg2) = repmat({'R'},length(gg2),1);

univar = numdata_out(:,[6 8 31 37 43 51 52 54 57 71 75 76 78 93 97 98 101]);
univar_label = textdata(1,[6 8 31 37 43 51 52 54 57 71 75 76 78 93 97 98 101]);

ylabels = {'Total distance (cm)', 'Time in center (sec)', 'Immobility time (sec)',... 
    'Climbing time (sec)','Swimming time (sec)','Social interaction ratio', 'Time in corners ratio',...
    'Time in open arms', 'Risk assessment (sec)', 'Discrimination index (short term trial)',...
    'Discrimination index (long term trial)', 'Amplitude of startle reflex (dB)',...
    'Prepulse inhibition (%)', 'Sucrose consumption', 'Average sucrose preference', 'Escape failures',...
    'Mean latency to escape (sec)'};

for i=1:size(univar,2)
    figure
    subplot(1,2,1)
    vp = violinplot(univar(:,i),groups_ISNS);
    vp(1).ViolinColor = [1 0 0]; % IS
    vp(2).ViolinColor = [0 0 1]; % NS
    ylabel(ylabels(i))
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')  

    subplot(1,2,2)
    vp = violinplot(univar(:,i),groups_HR);
    vp(1).ViolinColor = [0.5 0.5 0.5]; % H
    vp(2).ViolinColor = [0.2 0.8 0.2]; % R
    set(gca,'fontname','helvetica','fontsize',12,'linewidth',1.5,'xcolor','k','ycolor','k','box','off')  

    set(gcf,'color','white', 'position', [300 200 770 340])

end
%print -dpdf -painters violinplot_SIratio

clearvars -except alldata numdata textdata numdata_out relevant_data relevant_data_label ...
    prindata prindata_label idx_kcluster univar univar_label ylabels

% Statistics

    %IS vs NS
    for i=1:size(univar,2)
    [h,~] = kstest2(univar(prindata(:,1)==1,i),univar(prindata(:,1)==0,i));
        if h==0
            [~,p,~,tstats] = ttest2(univar(prindata(:,1)==1,i),univar(prindata(:,1)==0,i));
            disp([ylabels{i} ': p=' num2str(p) ', tstat: ' num2str(tstats.tstat) ', df: ' num2str(tstats.df) ', sd: ' num2str(tstats.sd)]);
        else
            [p,~,wstats] = ranksum(univar(prindata(:,1)==1,i),univar(prindata(:,1)==0,i));
            disp([ylabels{i} ': p=' num2str(p) ', zval: ' num2str(wstats.zval) ', ranksum: ' num2str(wstats.ranksum)]);
        end
    clear h p tstats wstats
    end

    %H vs R
    for i=1:size(univar,2)
    [h,~] = kstest2(univar(idx_kcluster(:,1)==1,i),univar(idx_kcluster(:,1)==2,i));
        if h==0
            [~,p,~,tstats] = ttest2(univar(idx_kcluster(:,1)==1,i),univar(idx_kcluster(:,1)==2,i));
            disp([ylabels{i} ': p=' num2str(p) ', tstat: ' num2str(tstats.tstat) ', df: ' num2str(tstats.df) ', sd: ' num2str(tstats.sd)]);
        else
            [p,~,wstats] = ranksum(univar(idx_kcluster(:,1)==1,i),univar(idx_kcluster(:,1)==2,i));
            disp([ylabels{i} ': p=' num2str(p) ', zval: ' num2str(wstats.zval) ', ranksum: ' num2str(wstats.ranksum)]);
        end
    clear h p tstats wstats
    end

    % T-test
    for i=1:size(univar,2)
    [~,p,~,tstats] = ttest2(univar(prindata(:,1)==1,i),univar(prindata(:,1)==0,i));
    disp([ylabels{i} ': p=' num2str(p) ', tstat: ' num2str(tstats.tstat) ', df: ' num2str(tstats.df) ', sd: ' num2str(tstats.sd)]);
    end
    for i=1:size(univar,2)
    [~,p,~,tstats] = ttest2(univar(idx_kcluster(:,1)==1,i),univar(idx_kcluster(:,1)==2,i));
    disp([ylabels{i} ': p=' num2str(p) ', tstat: ' num2str(tstats.tstat) ', df: ' num2str(tstats.df)  ', sd: ' num2str(tstats.sd)]);
    end

    % Levene's test
    for i=1:size(univar,2)
        [p,stats]=vartestn(univar(:,i),groups_ISNS,'TestType','LeveneAbsolute')
    end

%% Studies using similar protocols

[ndata, ~, alldata] = xlsread('Studies_similar_protocols.xlsx');

[idx_row, idx_col]=find(isnan(ndata));

OF = mean(ndata(1:3,:),1,'omitnan');
FS = mean(ndata(4,:),1,'omitnan');
SoP = mean(ndata(5:10,:),1,'omitnan');
EPM = mean(ndata(6:9,:),1,'omitnan');
SuP = mean(ndata(10,:),1),'omitnan';
EF = mean(ndata(11:14,:),1,'omitnan');
%data={OF,FS,SoP,EPM,Start,SuP,EF};
data=[OF;FS;SoP;EPM;SuP;EF];
data(isnan(data))=0;
[lin, col]=find(data==0)


data(1,:)=data(1,:)*3
data(2,:)=data(2,:)*1
data(3,:)=data(3,:)*1
data(4,:)=data(4,:)*4
data(5,:)=data(5,:)*1
data(6,:)=data(6,:)*4


lin=[1 4 3 2 5 6];
for i=1:6
    datax(lin(i),:)=data(i,:);
end

for i=1:6
datax(i,:)=smoothdata(datax(i,:),'SmoothingFactor',0.25);
end
figure, imagesc(datax), colormap redblue, c=colorbar;

%for i=1:41
%data(lin(i),col(i))=0;
%end
%figure, imagesc(data), colormap redblue, c=colorbar;

label = ({'1 day', '2 days','3 days','<6 days','1 week','2 weeks','3 weeks','4 weeks'})
set(gca,'XTick',1:8,'XTickLabel',label,'XTickLabelRotation',-45)

labely=({'OF','FS','SoP','EPM','SuP','EF'});
set(gca,'YTick',1:6,'YTickLabel',labely(lin))

set(gca,'fontname','helvetica','fontsize',10,'linewidth',1.5,'xcolor','k','ycolor','k','box','on')
set(gcf,'color','white')

