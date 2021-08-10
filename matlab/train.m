clc; clear; close all;
run("plot_config.m")
%%

SESSION_ID = "train_sim";

QTABLE  = DIR_DATA + "/" + SESSION_ID + "/qtable.csv";
CUM_R   = DIR_DATA + "/" + SESSION_ID + "/cum_reward.csv";
WL_AVG  = DIR_DATA + "/" + SESSION_ID + "/wl_avg.csv";
QDELTA  = DIR_DATA + "/" + SESSION_ID + "/qtable_diff.csv";
EPSILON = DIR_DATA + "/" + SESSION_ID + "/epsilon.csv";


figure
set(gcf, 'Position', [0 0 1300 500]);
t = tiledlayout(1,3);
% t.TileSpacing = 'compact';
t.Padding = 'compact';

% cumulative reward

data = readmatrix(CUM_R);
data_filtered = smoothdata(data, 'gaussian', 100);

% figure()
nexttile
pbaspect([1 1 1])
hold on
plot(data(:, 1), data(:, 2), "LineWidth", 0.1, "Color", COL_LIGHTGRAY);
plot(data_filtered(:, 1), data_filtered(:, 2), "LineWidth", 2);

title("Cumulative reward")
xlabel("Episode")
ylabel("Cumulative reward")
ax = gca;
ax.XRuler.Exponent = 0;

% win-loss moving average

data = readmatrix(WL_AVG);
data_filtered = smoothdata(data, 'gaussian', 100);

% figure()
nexttile
pbaspect([1 1 1])
hold on
plot(data(:, 1), data(:, 2), "LineWidth", 0.1, "Color", COL_LIGHTGRAY);
plot(data_filtered(:, 1), data_filtered(:, 2), "LineWidth", 2);

title("Moving average W/L")
xlabel("Episode")
ylabel("W/L ratio")
ax = gca;
ax.XRuler.Exponent = 0;

% Q converge

data = readmatrix(QDELTA);
data_filtered = smoothdata(data, 'gaussian', 100);

% figure()
nexttile
pbaspect([1 1 1])
hold on
plot(data(:, 1), data(:, 2), "LineWidth", 0.1, "Color", COL_LIGHTGRAY);
plot(data_filtered(:, 1), data_filtered(:, 2), "LineWidth", 2);

title("Difference in Q-values")
xlabel("Episode")
ylabel("Difference in Q-values")
ax = gca;
ax.XRuler.Exponent = 0;

legend("Raw", "Filtered")

export_fig(DIR_IMGS + "/" + SESSION_ID + ".pdf")
