clc; clear; close all;
run("plot_config.m")
%% Q-tables

STATES = {'HOME', 'HOME\_CAN\_KILL', 'GLOBE', 'GLOBE\_CAN\_KILL', 'GLOBE\_IN\_DANGER', 'GLOBE\_IN\_DANGER\_CAN\_KILL', 'COMMON\_PATH', 'COMMON\_PATH\_CAN\_KILL', 'COMMON\_PATH\_IN\_DANGER', 'COMMON\_PATH\_IN\_DANGER\_CAN\_KILL', 'COMMON\_PATH\_WITH\_BUDDY', 'COMMON\_PATH\_WITH\_BUDDY\_CAN\_KILL', 'VICTORY\_ROAD', 'GOAL'};
ACTIONS = {'MOVE\_FROM\_HOME', 'MOVE\_FROM\_HOME\_AND\_KILL', 'MOVE', 'MOVE\_ONTO\_STAR', 'MOVE\_ONTO\_STAR\_AND\_DIE', 'MOVE\_ONTO\_STAR\_AND\_KILL', 'MOVE\_ONTO\_GLOBE', 'MOVE\_ONTO\_GLOBE\_AND\_DIE', 'MOVE\_ONTO\_ANOTHER\_DIE', 'MOVE\_ONTO\_ANOTHER\_KILL', 'MOVE\_ONTO\_VICTORY\_ROAD', 'MOVE\_ONTO\_GOAL', 'NONE'};

SESSION_ID = "train_adv";
data = readmatrix(DIR_DATA + "/" + SESSION_ID + "/qtable.csv");
% data = normalize(data, 'norm');

figure
set(gcf, 'Position', [0 0 1000 700]);
h = heatmap(ACTIONS, STATES, data);
hxp = struct(h);
hxp.Axes.XAxisLocation = 'top';

% colormap(white) % white for empty, 
% h.ColorbarVisible = 'off';

export_fig(DIR_IMGS + "/qtable_" + SESSION_ID + ".pdf")

%% W/L tests

figure
set(gcf, 'Position', [0 0 1300 500]);
t = tiledlayout(1,3);
% t.TileSpacing = 'compact';
t.Padding = 'compact';

% for str = ["test_train_sim","test_train_adv", "test_ql_ql"]
% 
%     data = readmatrix(DIR_DATA + "/" + str + "/wl.csv");
% 
%     nexttile
%     h = histfit(data(:,2), 10)
%     pd = fitdist(data(:,2), "Normal")
% 
%     % title("W/L ratio of random player")
%     pbaspect([1 1 1])
%     yt = get(gca, 'YTick');
%     set(gca, 'YTick', yt, 'YTickLabel', yt/numel(data(:,2)))
%     xlabel("W/L ratio"); ylabel("Probability");
%     set(h(1),'facecolor', COL_LIGHTGRAY); set(h(2),'color', MATLAB_COLORS{2})
%   
% end

% simple vs random
data = readmatrix(DIR_DATA + "/test_train_sim/wl.csv");

nexttile
h = histfit(data(:,2), 10)
pd = fitdist(data(:,2), "Normal")

title("Simple vs Random")
pbaspect([1 1 1])
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(data(:,2)))
xlabel("W/L ratio"); ylabel("Probability");
set(h(1),'facecolor', COL_LIGHTGRAY); set(h(2),'color', MATLAB_COLORS{2})

a = annotation('textbox',[0.069 + 0 0.712 0.1 0.1]);
set(a,'String',{ sprintf('Mean: %3.3f', pd.mu), sprintf('Std Deviation: %3.3f', pd.sigma) });

% complex vs random
data = readmatrix(DIR_DATA + "/test_train_adv/wl.csv");

nexttile
h = histfit(data(:,2), 10)
pd = fitdist(data(:,2), "Normal")

title("Complex vs Random")
pbaspect([1 1 1])
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(data(:,2)))
xlabel("W/L ratio"); ylabel("Probability");
set(h(1),'facecolor', COL_LIGHTGRAY); set(h(2),'color', MATLAB_COLORS{2})

a = annotation('textbox',[0.069 + 0.32 0.712 0.1 0.1]);
set(a,'String',{ sprintf('Mean: %3.3f', pd.mu), sprintf('Std Deviation: %3.3f', pd.sigma) });

% simple vs random
data = readmatrix(DIR_DATA + "/test_ql_ql/wl.csv");

nexttile
h = histfit(data(:,2), 12)
pd1 = fitdist(data(:,2), "Normal")
set(h(1),'facecolor', COL_PINK ); set(h(2),'color', MATLAB_COLORS{2})

hold on

h = histfit(data(:,3), 12)
pd2 = fitdist(data(:,3), "Normal")
set(h(1),'facecolor', COL_MINT); set(h(2),'color', MATLAB_COLORS{2})

title("Simple vs Complex")
pbaspect([1 1 1])
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(data(:,2)))
xlabel("W/L ratio"); ylabel("Probability");
legend("Simple", '', "Complex", '')

a = annotation('textbox',[0.069 + 0.641 0.712 0.1 0.1]);
set(a,'String',{ sprintf('Mean: %3.3f, %3.3f', pd1.mu, pd2.mu), sprintf('Std Deviation: %3.3f, %3.3f', pd1.sigma, pd2.sigma) });

export_fig(DIR_IMGS + "/test_ql.pdf")
