function [dist] = reach_distance(DataSet,p,o,k)
    %pµÄ¿É´ï¾àÀë
    %pÊÇÖ÷½Úµã£¬oÈÏÎªÊÇpµÄÁÚ¾Ó£¬ËÑË÷·¶Î§Îªk
    k_dist = DDOutlier.k_distance(DataSet,o,k);
    %k_dist
    dist = DDOutlier.distance(DataSet,p,o,k);
    %dist
    dist = max(k_dist,dist);
    %disp(dist)
end