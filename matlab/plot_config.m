% PLEASE LOAD THIS USING:
% run("plot_config.m")

% directories
DIR_CURRENT    = erase(strrep(which(mfilename),"\","/"), "/" + mfilename + ".m");
DIR_EXPORT_FIG = DIR_CURRENT + "/export_fig";
DIR_DATA       = DIR_CURRENT + "/../data";
DIR_IMGS       = DIR_CURRENT + "/../assets/img";

% configure export_fig
% https://github.com/altmany/export_fig
addpath(DIR_EXPORT_FIG);

% configure figure defaults
set(groot, "DefaultFigureRenderer", "painters");
set(groot, "DefaultFigurePosition", [0 0 500 500]);
set(groot, "DefaultFigureColor", [1 1 1]);
set(groot, "DefaultAxesFontSize", 12); % !!!

% colors
COL_GRAY = [200 200 200]/255;
COL_LIGHTGRAY = [220 220 220]/255;
COL_ORANGE = [255 143 0]/255;
COL_BLUE = [0 207 255]/255;
COL_MAGENTA = [236 88 234]/255;
COL_MINT = [131 240 220]/255;
COL_ROSE = [240, 131, 136]/255;
COL_PINK = [240, 158, 189]/255;

% default MATLAB colors
% http://math.loyola.edu/~loberbro/matlab/html/colorsInMatlab.html
MATLAB_COLORS = {
    [0.0000, 0.4470, 0.7410] ;
    [0.8500, 0.3250, 0.0980] ;
    [0.9290, 0.6940, 0.1250] ;
    [0.4940, 0.1840, 0.5560] ;
    [0.4660, 0.6740, 0.1880] ;
    [0.3010, 0.7450, 0.9330] ;
    [0.6350, 0.0780, 0.1840] ;
};
