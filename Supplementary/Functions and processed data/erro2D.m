% ERRO2D() - calcula erro padrão da média de vetor amostra
%   Inputs  =   dado(observações,amostras)
%               dim: dimensão para cálculo. 
%                   1=linhas
%                   2=colunas   [default: maior dimensão]
%   Output  =   erro padrão
%
% Autor: Danilo Benette Marques, 2016

function [erro_padrao]=erro2D(dado,dim);
if nargin < 2
    [~,dim]=max(size(dado));
    erro_padrao=nanstd(dado,0,dim)/sqrt(size(dado,dim));
elseif nargin > 1
    erro_padrao=nanstd(dado,0,dim)/sqrt(size(dado,dim));
%OBS: std(x,0,dim) == normalização por N-1
end
