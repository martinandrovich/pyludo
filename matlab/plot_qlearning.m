clc; clear; close all;
run("plot_config.m")

SESSION_ID = "train_6";

QTABLE = DIR_DATA + "/" + SESSION_ID + "/qtable.csv";
WL_AVG = DIR_DATA + "/" + SESSION_ID + "/wl_avg.csv";
QDELTA = DIR_DATA + "/" + SESSION_ID + "/qtable_diff.csv";

%% heatmap of Q-table

data = readmatrix(QTABLE);

states = {'HOME', 'HOME\_CAN\_KILL', 'GLOBE', 'GLOBE\_CAN\_KILL', 'GLOBE\_IN\_DANGER', 'GLOBE\_IN\_DANGER\_CAN\_KILL', 'COMMON\_PATH', 'COMMON\_PATH\_CAN\_KILL', 'COMMON\_PATH\_IN\_DANGER', 'COMMON\_PATH\_IN\_DANGER\_CAN\_KILL', 'COMMON\_PATH\_WITH\_BUDDY', 'COMMON\_PATH\_WITH\_BUDDY\_CAN\_KILL', 'VICTORY\_ROAD', 'GOAL'};
actions = {'MOVE\_FROM\_HOME', 'MOVE\_FROM\_HOME\_AND\_KILL', 'MOVE', 'MOVE\_ONTO\_STAR', 'MOVE\_ONTO\_STAR\_AND\_DIE', 'MOVE\_ONTO\_STAR\_AND\_KILL', 'MOVE\_ONTO\_GLOBE', 'MOVE\_ONTO\_GLOBE\_AND\_DIE', 'MOVE\_ONTO\_ANOTHER\_DIE', 'MOVE\_ONTO\_ANOTHER\_KILL', 'MOVE\_ONTO\_VICTORY\_ROAD', 'MOVE\_ONTO\_GOAL', 'NONE'};

figure
set(gcf, 'Position', [0 0 1500 1500]);
h = heatmap(actions, states, data);
hxp = struct(h);
hxp.Axes.XAxisLocation = 'top';

return

while true
    data = readmatrix(QTABLE);
    h.ColorData = data;
    drawnow;
end

%%

data = readmatrix(WL_AVG);
data_filtered = smoothdata(data);

figure()
plot(data(:, 1), data(:, 2));

ax = gca;
ax.XRuler.Exponent = 0;

%%

data = readmatrix(QDELTA);
data_filtered = smoothdata(data, 'gaussian', 100);

figure()
hold on
plot(data(:, 1), data(:, 2));
plot(data_filtered(:, 1), data_filtered(:, 2));

ax = gca;
ax.XRuler.Exponent = 0;
