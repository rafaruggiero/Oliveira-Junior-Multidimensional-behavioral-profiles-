function [dist] = distance(DataSet,i,j,k)
    %ÕÒ¸öÌåiµ½ÁÚ¾ÓjµÄ¾àÀë¡£ËÑË÷·¶Î§ÊÇk
    %Èç¹ûÔÚk·¶Î§ÄÚ£¬j²»ÊÇiµÄÁÚ¾Ó¾Í»á³ö´í¡£
    [kdist_obj,~] = DDOutlier.kDistObj(DataSet,k);
    
    [~,neighborLevel_j] = find(kdist_obj.id(i,:) == j);
    if ~isempty(neighborLevel_j)
        dist = kdist_obj.dist(i,neighborLevel_j); 
    else
        disp("iµÄkÁÚ¾ÓÀïÃæÃ»ÓÐj!");
        [~,neighborLevel_i] = find(kdist_obj.id(j,:) == i);
        if ~isempty(neighborLevel_i)
            dist = kdist_obj.dist(j,neighborLevel_i); 
        else
            error("iºÍj²»ÔÚ¸÷×ÔµÄkÁÚ¾Ó·¶Î§ÄÚ£¡");
        end
    end
    
end