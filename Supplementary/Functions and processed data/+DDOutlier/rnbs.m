function [Rnbi,numb] = rnbs(DataSet,k)
    %rnbs ÎªÃ¿Ò»¸öµãÕÒµ½°Ñ×Ô¼ºµ±ÁÚ¾ÓµÄÆäËûµãµÄ¸öÊý¡£
    %Rnbi ¾ÍÊÇÃ¿Ò»¸öµãµÄ»¶Ó­¶È
    %numb ÊÇ²»ÊÜ»¶Ó­¸öÌåµÄ¸öÊý
    
    if k > DataSet.nn
        DataSet.increaseBuffer(k + 10);
    end
    [kdist_obj,~] = DDOutlier.kDistObj(DataSet,k);
    
    
    edges = [0.5:1:(DataSet.n + 0.5)];
    [Rnbi,~] = histcounts(kdist_obj.id,edges);
    Rnbi = Rnbi';
    
    numb = sum(Rnbi == 0);
end