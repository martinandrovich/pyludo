clc; clear; close all;
run("plot_config.m")
%%

avg = [];

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
end


x = categorical({'Simple','Complex'});
x = reordercats(x, {'Simple','Complex'});

figure
set(gcf, 'Position', [0 0 1000 500]);
b = bar(x, avg, 0.6)
xtips1 = b(1).XEndPoints;
ytips1 = b(1).YEndPoints;
labels1 = string(b(1).YData);
text(xtips1,ytips1,labels1,'HorizontalAlignment','center',...
    'VerticalAlignment','bottom')

title("Q-learning vs Expert players")
legend("Q-Learning", "Fast", "Aggresive", "Defensive")
xlabel("State-space representation"); ylabel("Average W/L ratio")

export_fig(DIR_IMGS + "/test_ql_std.pdf")