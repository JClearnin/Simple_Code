
% 数据
hellobaby = rand(1,50);


% 创建一个图形F
figure;

% 绘制准确率曲线
plot(hellobaby, '*-', 'Color', [0.85 0.33 0.10], 'MarkerSize', 8, 'LineWidth', 1.8, 'DisplayName', 'hellobaby');

% 标出最大值点
[max_baby, max_baby_idx] = max(hellobaby);

text(max_baby_idx, max_baby, sprintf('Baby Max: %.2f%%', max_baby * 100), 'Color', 'black', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right','FontSize',15, 'FontName','Times New Roman');

% 添加标签和标题
xlabel('Baby-x','FontName','Times New Roman','FontSize', 16.5);
ylabel('Baby-y','FontName','Times New Roman','FontSize', 16.5);

xticks(0:5:length(hellobaby)); 
yticks(0:0.1:1);  % 从 0 到 1，间隔为 0.1
ylim([0 1.05])
set(gca,'FontName','Times New Roman','FontSize', 15)
% 添加图例
legend('Location', 'Best','FontName','Times New Roman','FontSize', 15);

% 显示网格
grid on;

% 创建放大的坐标轴
ax2 = axes('Position', [.10 .6 .25 .25]);
box on;

% 绘制放大的数据
plot(ax2, sever_acc_fig, '*-', 'Color', '[0.23 0.43 0.10]', 'MarkerSize', 8, 'LineWidth', 1.8, 'DisplayName', 'hello-baby');

xticks(ax2, 0:5:25); 
yticks(ax2, 0:0.1:1); 


% xlim([11.5 13.5])
xlim([20 25.5])

ylim([0 1])
% 显示网格
grid(ax2, 'on');
