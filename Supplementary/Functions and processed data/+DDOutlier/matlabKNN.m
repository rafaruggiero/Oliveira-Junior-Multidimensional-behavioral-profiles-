function [id,dist] = matlabKNN(data,k,disMetric)
%matlabKNN ÊÇR°üDBSCANµÄKNNº¯ÊýµÄMATLAB¼òÒªÊµÏÖ
%   
    %¶¨Òå¾àÀëº¯ÊýÀ´Ô´
    if strcmp(disMetric,'euclidean')
        disMetric = 'euclidean';
    else
        error("Î´ÖªµÄ¾àÀë²ÎÊý");
    end
    
    %matlabÔËÐÐÐèÒªµÄkÒª¶à¼ÓÒ»¸ö
    k = k + 1;
    
    [n,~] = size(data);
    id = zeros(n,k);
    dist = zeros(n,k);
    parfor i = 1:1:n
        datum = data(i,:);
        %[aID,aDist] = knnsearch(data,datum,'K',k,'Distance',disMetric,...
        %'NSMethod','kdtree');
        [aID,aDist] = knnsearch(data,datum,'K',k,...
            'Distance',disMetric,'IncludeTies',true);
        aID = aID{1}(1:(k));
        aDist = aDist{1}(1:(k));
        id(i,:) = aID;
        dist(i,:) = aDist;
    end
    
    %²Ã¼ô½á¹ûÒÔÆ¥ÅäÊä³ö
    id1 = id(:,1);
    dist1 = dist(:,1);
    
    id = id(:,2:end);
    dist = dist(:,2:end);
    
    %¼ì²é½á¹û·ÀÖ¹³öÏÖ×Ô¼º(ÓÐµÄÊ±ºò£¬×Ô¼º²»ÊÇ±»ÅÅÔÚµÚÒ»¸ö£¬ËùÒÔÒ»µ¶¼õÏÂÈ¥²»ÊÇÌØ±ð¶Ô)
    for i = 1:1:n
        mySelf = find(id(i,:) == i);
        if(~isempty(mySelf))
            %fprintf("ÔªËØ%dÔÚ%dÁÐ±»´íÎó±àÈë£¡\n",i,mySelf);
            id(i,mySelf) = id1(i);
            dist(i,mySelf) = dist1(i);
        end
    end
end

