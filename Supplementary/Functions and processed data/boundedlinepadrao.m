%BOUNDEDLINEPADRAO() - Plot bounded line com média e erro de matriz
% 
% Inputs:
%     x = vetor do eixo x [aceita x=[] ]
%     y = matriz de dados
%     cor = cor do plot
%     dim = dimensão para cálculo da média e erro
%     holdon = figure=0, hold on=1 [default=0]
%     
function [h1,patch1]=boundedlinepadrao(x,y,cor,dim,holdon);

if nargin<5
    holdon = 0;
end

if isempty(x)==1
    if dim==1
        dim2=2;
    elseif dim==2
        dim2=1;
    end
    x = 1:size(y,dim2);
end

color=repmat(cor,length(x),1);

if holdon==0
    figure
elseif holdon==1
    hold on
end

x = shiftdim(x);

meany = shiftdim( nanmean(y,dim));
erroy = shiftdim( erro2D(y,dim));

[h1,patch1]=boundedline(x,meany,erroy,'cmap',color,'alpha'); set(h1,'linewidth',2)

hold off

end