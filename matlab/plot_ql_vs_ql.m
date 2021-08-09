clc; clear; close all;
run("plot_config.m")

TEST_ID = "test_ql_ql/test_11";
WL_DIST = DIR_DATA + "/" + TEST_ID    + "/wl.csv";

%%

data = readmatrix(WL_DIST);
wl_sim = data(:,2);
wl_adv = data(:,3);

wl = [wl_sim wl_adv];
figure
normplot(wl)

figure
histogram(wl_sim)
hold on
histogram(wl_adv)

[h,p,ci,stats] = ttest2(wl_sim, wl_adv)

% h = histfit(wl_sim)
% pd = fitdist(wl_sim, "Normal")
