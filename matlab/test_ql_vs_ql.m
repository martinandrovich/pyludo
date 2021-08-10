clc; clear; close all;
run("plot_config.m")
%%
close all;
data = readmatrix(DIR_DATA + "/test_ql_ql/wl.csv");
wl_sim = data(:,2);
wl_adv = data(:,3);

wl = [wl_sim wl_adv];
figure
normplot(wl)

figure
histogram(wl_sim)
hold on
histogram(wl_adv)

% tail left = mean(x) < mean(y)
[h,p,ci,stats] = ttest2(wl_sim, wl_adv)

% h = histfit(wl_sim)
% pd = fitdist(wl_sim, "Normal")
