% SORTK() - Sort indices of cluster ID numerical vector by sum of clustered data
% 
%   Usage:
%       [idxK] = sortk(idxK,data)
% 
%   Inputs
%       idxK = vector of cluster ID (number)
%       data = data to sort by sum
% 
%   Outputs
%       idxK = sorted vector of cluster ID
% 
% Author: Danilo Benette Marques, 2023

function [idxK] = sortk(idxK,data)

%Number of clusters
k = numel(unique(idxK));
%Clusters ID
idxk = unique(idxK);

%Average of cluster data
for ik = 1:k
    avgclust(ik,:) = nanmean(data(idxK==idxk(ik),:),1);
end

%Sum of cluster average
sumavgclust = sum(avgclust,2);
%Sort indices
[~,isort] = sort(sumavgclust); 

%Replace cluster ID by sort index
idxK = idxK+100;
for ik = 1:k
    idxK(idxK==100+isort(ik)) = ik;
end


end