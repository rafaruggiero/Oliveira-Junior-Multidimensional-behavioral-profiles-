%% Generate colormap [red; blue]

function cmap = colormaprb

cmap(:,1) = linspace(0,1,64);
cmap(:,2) = linspace(0,0,64);
cmap(:,3) = linspace(1,0,64); 

end