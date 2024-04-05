function [r,max_nb] = NaNSearching(DataSet)
    %ÀûÓÃNatural Neighbor searchingÀ´ÕÒµ½
    %×ÔÊÊÓ¦µÄËÑË÷°ë¾¶rºÍ×îÊÜ»¶Ó­µÄµãµÄÁÚ¾ÓÊýmax_nb
    r = 1;
    %×ÔÊÊÓ¦Ñ°ÕÒËÑË÷·¶Î§
    while r <= DataSet.n
        %×ÔÊÊÓ¦ËÑË÷·¶Î§
        fprintf("r is now:%d\n",r);

        [Rnbi,numb] = DDOutlier.rnbs(DataSet,r);
        if r == 1
            %ÈçÊµÕâÊÇµÚÒ»´ÎÑ­»·£¬¾Í³õÊ¼»¯ÉÏÒ»´ÎËÑË÷°ë¾¶
            numb_upd = numb;
            r = r + 1;
        elseif numb_upd == numb
            break;
        else
            numb_upd = numb;
            r = r + 1;
        end

    end

    %×îÊÜ»¶Ó­µãµÄ»¶Ó­¶È
    max_nb = max(Rnbi);
end