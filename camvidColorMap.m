function cmap = camvidColorMap()
% Define the colormap used by CamVid dataset.

cmap = [
    180, 180, 235;   % Darker light blue
    235, 0, 0;   % Darker light red
    200, 200, 200;   % Darker light gray
    0, 235, 180;   % Darker light green
    235, 210, 0;   % Darker light orange
    100, 10, 200;   % Darker light lime
    235, 180, 235;   % Darker light magenta
    180, 235, 235;   % Darker light cyan
    235, 0, 180;   % Darker light yellow
    180, 180, 180;   % Darker light gray
    235, 130, 130;   % Darker light pink
    ];


% Normalize between [0 1].
cmap = cmap ./ 255;
end