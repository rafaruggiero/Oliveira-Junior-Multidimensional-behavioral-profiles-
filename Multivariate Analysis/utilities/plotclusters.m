% PLOTCLUSTERS() - Plot observations colored per cluster
%   Usage
%       [h] = plotclusters(X,idxK,avgorall)
% 
%   Inputs 
%       X   = data (obs,vars)
%       idxK = index of clusters (ID vector)
%       avgorall = plot obs or average
%               0: observations [default]
%               1: average
% 
%   Output 
%       h = handles 
% 
% Autor: Danilo Benette Marques, 2022

function [h] = plotclusters(X,idxK,avgorall)

if nargin<3
    avgorall = 0;
end

cmap = colormap('lines');

idxk = unique(idxK);

if avgorall == 0 %plot observations
    for ik = 1:numel(idxk)
        hold on
        H{ik} = plot(X(idxK==idxk(ik),:),'color',cmap(ik,:));
    end
    
elseif avgorall == 1 %plot average
%     hold on,plot(X,'color',[.8 .8 .8])
    for ik = 1:numel(idxk)
        hold on
        try %has boundedlinepadrao boundedline
        H{ik} = boundedlinepadrao([],X(idxK==idxk(ik),:),cmap(ik,:),1,1);
        catch
        H{ik} = plot(nanmean(X(idxK==idxk(ik),:),1) ,'color',cmap(ik,:), 'linewidth',2);
        end
    end
    
end


end