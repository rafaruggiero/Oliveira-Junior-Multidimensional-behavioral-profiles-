%% Generate colormap [blue; white; red]

function cmap = colormaprwb

cmapb = zeros(31,3);
cmapb(:,1) = linspace(0,1,31);
cmapb(:,2) = linspace(0,1,31);
cmapb(:,3) = linspace(1,1,31); %blue

cmapw = [1 1 1];

cmapr = zeros(31,3);
cmapr(:,1) = linspace(1,1,31); %red
cmapr(:,2) = linspace(1,0,31);
cmapr(:,3) = linspace(1,0,31);

cmap = [cmapb; cmapw; cmapr];
end