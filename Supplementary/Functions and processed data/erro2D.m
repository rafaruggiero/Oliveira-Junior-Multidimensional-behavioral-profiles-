% ERRO2D() - calcula erro padr�o da m�dia de vetor amostra
%   Inputs  =   dado(observa��es,amostras)
%               dim: dimens�o para c�lculo. 
%                   1=linhas
%                   2=colunas   [default: maior dimens�o]
%   Output  =   erro padr�o
%
% Autor: Danilo Benette Marques, 2016

function [erro_padrao]=erro2D(dado,dim);
if nargin < 2
    [~,dim]=max(size(dado));
    erro_padrao=nanstd(dado,0,dim)/sqrt(size(dado,dim));
elseif nargin > 1
    erro_padrao=nanstd(dado,0,dim)/sqrt(size(dado,dim));
%OBS: std(x,0,dim) == normaliza��o por N-1
end
