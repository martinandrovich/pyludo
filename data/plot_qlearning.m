clc; clear; close all;
format longG

SESSION_ID = 1;
QTABLE = SESSION_ID + "/qtable.csv";
WL_AVG = SESSION_ID + "/wl_avg.csv";
QDELTA = SESSION_ID + "/qtable_diff.csv";

%%

close all;

data = readmatrix(QTABLE);

states = {'HOME', 'GLOBE', 'STAR', 'STAR\_IN\_DANGER', 'COMMON\_PATH', 'COMMON\_PATH\_WITH\_BUDDY', 'COMMON\_PATH\_CAN\_KILL', 'COMMON\_PATH\_IN\_DANGER', 'VICTORY\_ROAD', 'GOAL'};
actions = {'MOVE\_FROM\_HOME', 'MOVE', 'MOVE\_ONTO\_STAR', 'MOVE\_ONTO\_STAR\_AND\_DIE', 'MOVE\_ONTO\_STAR\_AND\_KILL', 'MOVE\_ONTO\_GLOBE', 'MOVE\_ONTO\_GLOBE\_AND\_DIE', 'MOVE\_ONTO\_ANOTHER\_DIE', 'MOVE\_ONTO\_ANOTHER\_KILL', 'MOVE\_ONTO\_VICTORY\_ROAD', 'MOVE\_ONTO\_GOAL', 'NONE'};

h = heatmap(actions, states, data);
hxp = struct(h);
hxp.Axes.XAxisLocation = 'top';

while true
    data = readmatrix(QTABLE);
    h.ColorData = data;
    drawnow;
end

%%

data = readmatrix(WL_AVG);
plot(data(:, 1), data(:, 2));

ax = gca;
ax.XRuler.Exponent = 0;

%%

data = readmatrix(QDELTA);
plot(data(:, 1), data(:, 2));

ax = gca;
ax.XRuler.Exponent = 0;