clc; clear; close all;
run("plot_config.m")
%%

data_wl = readmatrix(DIR_DATA + "/test_rng/wl.csv");

figure
h = histfit(data_wl(:,2))
pd = fitdist(data_wl(:,2), "Normal")
pbaspect([1 1 1])

title("W/L ratio of a random player")
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(data_wl(:,2)))
xlabel("W/L ratio"); ylabel("Probability");
set(h(1),'facecolor', COL_LIGHTGRAY); set(h(2),'color', MATLAB_COLORS{2})

a = annotation('textbox',[0.151 0.781 0.1 0.1]);
set(a,'String',{ sprintf('Mean: %3.3f', pd.mu), sprintf('Std Deviation: %3.3f', pd.sigma) });

export_fig(DIR_IMGS + "/test_rng.pdf")
