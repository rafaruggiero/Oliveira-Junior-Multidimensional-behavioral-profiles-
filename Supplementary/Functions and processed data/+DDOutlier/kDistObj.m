function [kdist_obj,increaseKs] = kDistObj(DataSet,k)
    % kDistObj Éú³ÉÒ»¸ö¾ØÕó£¬Õâ¸ö¾ØÕóÄÒÀ¨ÁË¸ÕºÃµÈÓÚk¾àÀëµÄÄÇÐ©ÁÚ¾Óµã¡£
    % ÓÉÓÚ²»Í¬µÄ½ÚµãkÁÚ¾Ó²»Í¬£¬ËùÒÔ»áµ¼ÖÂ³¤¶Ì²»Ò»¡£ÎªÁËÈÔÈ»ÄÜ·ñ´æ´¢ÔÚÒ»¸ö¾ØÕóÀï¡£
    % ÓÃÎÞÐ§µÄÔªËØ²¹È«³¤¶Ì²»ÆëµÄÔªËØ¡£
    % ÓÃÓÚ²¹ÆëµÄ¾àÀëµÄÎÞÐ§ÔªËØÊÇinf
    % ÓÃÓÚ²¹ÆëµÄIDµÄÎÞÐ§ÔªËØÊÇ-1
    
    persistent k_buff;
    persistent kdist_obj_buff;
    persistent increaseKs_buff;
    
    if isempty(k_buff) || (k_buff ~= k)
        
        increaseKs = ones(1,DataSet.n) * k;
        for i = 1:1:DataSet.n
            while DDOutlier.k_distance(DataSet,i,increaseKs(i)+1) <= ...
                    DDOutlier.k_distance(DataSet,i,increaseKs(i))
                increaseKs(i) = increaseKs(i) + 1;
                %warning("·¢ÏÖ¾àÀëÏàµÈµÄÔªËØ¡£");
            end
            %fprintf("ÐÐ%dÀ©Õ¹µ½£º%d\n",i,increaseKs(i));
        end
        increaseKsMAX = max(increaseKs);
        kdist_obj = struct();
        kdist_obj.dist = zeros(DataSet.n,increaseKsMAX);
        kdist_obj.id = zeros(DataSet.n,increaseKsMAX);
        buffdist = kdist_obj.dist;
        buffid = kdist_obj.id;
        parfor i = 1:DataSet.n
            buffdist(i,:) = ...
                [DataSet.dist_obj.dist(i,1:increaseKs(i)) ...
                ones(1,increaseKsMAX-increaseKs(i))*inf];
            buffid(i,:) = ...
                [DataSet.dist_obj.id(i,1:increaseKs(i)) ...
                ones(1,increaseKsMAX-increaseKs(i))*(-1)];
        end
        kdist_obj.dist = buffdist;
        kdist_obj.id = buffid;
        
        %»º³å²»¶Ô£¬ÖØ½¨»º³å
        k_buff = k;
        kdist_obj_buff = kdist_obj;
        increaseKs_buff = increaseKs;
    else
        kdist_obj = kdist_obj_buff;
        increaseKs = increaseKs_buff;
    end
end
