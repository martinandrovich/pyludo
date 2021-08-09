% clc; clear; close all;
% run("plot_config.m")

SESSION_ID = "train_11_2p";
TEST_ID    = "test_ql_ql/test_11";

QTABLE  = DIR_DATA + "/" + SESSION_ID + "/qtable.csv";
CUM_R   = DIR_DATA + "/" + SESSION_ID + "/cum_reward.csv";
WL_AVG  = DIR_DATA + "/" + SESSION_ID + "/wl_avg.csv";
QDELTA  = DIR_DATA + "/" + SESSION_ID + "/qtable_diff.csv";
EPSILON = DIR_DATA + "/" + SESSION_ID + "/epsilon.csv";
WL_DIST = DIR_DATA + "/" + TEST_ID    + "/wl.csv";

%% Q-table

% close all;

data = readmatrix(QTABLE);
% data = normalize(data, 'norm');

states = {'HOME', 'HOME\_CAN\_KILL', 'GLOBE', 'GLOBE\_CAN\_KILL', 'GLOBE\_IN\_DANGER', 'GLOBE\_IN\_DANGER\_CAN\_KILL', 'COMMON\_PATH', 'COMMON\_PATH\_CAN\_KILL', 'COMMON\_PATH\_IN\_DANGER', 'COMMON\_PATH\_IN\_DANGER\_CAN\_KILL', 'COMMON\_PATH\_WITH\_BUDDY', 'COMMON\_PATH\_WITH\_BUDDY\_CAN\_KILL', 'VICTORY\_ROAD', 'GOAL'};
actions = {'MOVE\_FROM\_HOME', 'MOVE\_FROM\_HOME\_AND\_KILL', 'MOVE', 'MOVE\_ONTO\_STAR', 'MOVE\_ONTO\_STAR\_AND\_DIE', 'MOVE\_ONTO\_STAR\_AND\_KILL', 'MOVE\_ONTO\_GLOBE', 'MOVE\_ONTO\_GLOBE\_AND\_DIE', 'MOVE\_ONTO\_ANOTHER\_DIE', 'MOVE\_ONTO\_ANOTHER\_KILL', 'MOVE\_ONTO\_VICTORY\_ROAD', 'MOVE\_ONTO\_GOAL', 'NONE'};

figure
set(gcf, 'Position', [0 0 1000 700]);
h = heatmap(actions, states, data);

% colormap(white) % white for empty, 
% h.ColorbarVisible = 'off';
hxp = struct(h);
hxp.Axes.XAxisLocation = 'top';

export_fig(DIR_IMGS + "/" + SESSION_ID + ".pdf")

%% cumulative reward

data = readmatrix(CUM_R);
data_filtered = smoothdata(data, 'movmean', 100);

figure()
hold on
plot(data(:, 1), data(:, 2));
plot(data_filtered(:, 1), data_filtered(:, 2));

xlabel("Episode")
ylabel("Cumulative reward")
ax = gca;
ax.XRuler.Exponent = 0;

%% win-loss moving average

data = readmatrix(WL_AVG);
data_filtered = smoothdata(data, 'gaussian', 100);

figure()
hold on
plot(data(:, 1), data(:, 2));
plot(data_filtered(:, 1), data_filtered(:, 2), "LineWidth", 2);

xlabel("Episode")
ylabel("Average W/L")
ax = gca;
ax.XRuler.Exponent = 0;

%% Q converge

data = readmatrix(QDELTA);
data_filtered = smoothdata(data, 'gaussian', 100);

figure()
hold on
plot(data(:, 1), data(:, 2));
plot(data_filtered(:, 1), data_filtered(:, 2), "LineWidth", 2);

xlabel("Episode")
ylabel("Difference in Q-values")
ax = gca;
ax.XRuler.Exponent = 0;

%% epsilon

data = readmatrix(EPSILON);

figure()
hold on
plot(data(:, 1), data(:, 2), "LineWidth", 2);

xlabel("Episode")
ylabel("Epsilon")
ax = gca;
ax.XRuler.Exponent = 0;

%% W/L distribution

data = readmatrix(WL_DIST);
wl_dist = data(:,2);

h = histfit(wl_dist)
pd = fitdist(wl_dist, "Normal")
