classdef dataSet < handle
    %dataSet ÓÃÀ´Ìá¹©Êý¾Ý±¾ÉíÖ±½ÓÏà¹ØµÄ»ù´¡·þÎñ
    %   
    
    properties
        data = [];
        %¾àÀë³ß¶È
        disMetric = "";
      
        n = 0; %¹Û²â¸öÊý
        n_var = 0; %Êý¾ÝÎ¬¶È£¬±äÁ¿¸öÊý
        
        nn = 0; %Êý¾Ý»º³åÁ¿
        dist_obj = struct(); %Êý¾Ý»º³å
    end
    
    methods
        function obj = dataSet(dataIn,disMetric)
            %dataSet ¹¹Ôì´ËÀàµÄÊµÀý
             obj.data = dataIn;
             [obj.n,obj.n_var] = size(obj.data);
             obj.disMetric = disMetric;
             %Ð´Èë³õÆÚÔ¤²âµÄ»º³åÁ¿
             obj.nn = ceil(sqrt(obj.n));
             %»º³å¾àÀë¾ØÕó
             [id,dist] = DDOutlier.matlabKNN(obj.data,obj.nn,obj.disMetric);
             obj.dist_obj.id = id;
             obj.dist_obj.dist = dist;
             
        end
        
        function [] = increaseBuffer(obj,nn)
            %increaseBuffer Ôö¼Ó»º³å
            %   ÓÃÀ´ÔÚµ±Ç°»º³å²»¹»ÓÃµÄÊ±ºòÔö¼Ó»º³å£¬
            %²¢´¦ÀíÓÉÓÚ»º³åÔö¼Ó¶øµ¼ÖÂÆäËû²ÎÊý±ä¶¯µÄÎÊÌâ
            obj.nn = nn;
            [id,dist] = DDOutlier.matlabKNN(obj.data,obj.nn,obj.disMetric);
            obj.dist_obj.id = id;
            obj.dist_obj.dist = dist;
        end
    end
end

