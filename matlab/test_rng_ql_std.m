clc; clear; close all;
run("plot_config.m")
%%

figure
set(gcf, 'Position', [0 0 1500 500]);
t = tiledlayout(1,3);
% t.TileSpacing = 'compact';
t.Padding = 'compact';

% RNG

data_wl = readmatrix(DIR_DATA + "/test_rng/wl.csv");

nexttile
h = histfit(data_wl(:,2))
pd = fitdist(data_wl(:,2), "Normal")
pbaspect([1 1 1])

title("Random player")
yt = get(gca, 'YTick');
set(gca, 'YTick', yt, 'YTickLabel', yt/numel(data_wl(:,2)))
xlabel("W/L ratio"); ylabel("Probability");
set(h(1),'facecolor', COL_LIGHTGRAY); set(h(2),'color', MATLAB_COLORS{2})

a = annotation('textbox',[0.061 0.781 0.1 0.1]);
set(a,'String',{ sprintf('Mean: %3.3f', pd.mu), sprintf('Std Deviation: %3.3f', pd.sigma) });

% QL vs std

avg = [];
dev = [];

for method = ["test_train_sim_std", "test_train_adv_std"]
    
    path = DIR_DATA + "/" + method + "/dist.csv";
    
    data = fileread(path);
    x = strsplit(data, '\n')';
    x = replace(x, "'", '"');
    x = string(x(1:end-1));
    x = extract(x, "{" + wildcardPattern + "}");

    stats = zeros(length(x), 4);
    for i = 1:length(x)
        dist = jsondecode(x(i));
        stats(i,:) = [dist.q_learning dist.fast dist.aggressive dist.defensive];
    end
    
%     avg = mean(stats)/1000
    avg = [avg ; mean(stats)/1000]
    dev = [dev ; std(stats)/1000]
end


x = categorical({'Simple','Complex'});
x = reordercats(x, {'Simple','Complex'});

nexttile(2, [1 2])
b = bar(x, avg, 0.4)
pbaspect([2 0.9 1])
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

title("Q-learning vs Expert players")
legend("Q-Learning", "Fast", "Aggresive", "Defensive")
xlabel("State-space representation"); ylabel("Average W/L ratio")


export_fig(DIR_IMGS + "/test_rng_ql_std.pdf")