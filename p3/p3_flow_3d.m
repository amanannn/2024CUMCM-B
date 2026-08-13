%% 问题3 组装流程 3D 示意图（MATLAB）
% 结构：8 零配件 → 3 半成品 → 成品
% 输出：p3/figures/fig4_flow_3d.png (300dpi)
clear; close all; clc
if ~exist(fullfile('p3', 'figures'), 'dir'); mkdir(fullfile('p3', 'figures')); end

% ---------- 节点位置 ----------
% 零配件（z=0 平面，圆周排列）
part_pos = zeros(8, 3);
for i = 1:8
    ang = (i - 1) / 8 * 2 * pi + pi / 8;
    part_pos(i, :) = [1.9 * cos(ang), 1.9 * sin(ang), 0];
end
% 半成品（z=1.3，三角排列）
semi_pos = [-1.1, -0.7, 1.3; 1.1, -0.7, 1.3; 0, 1.3, 1.3];
% 成品（z=2.6，顶部中央）
final_pos = [0, 0, 2.6];

% ---------- 组装关系 ----------
semi_of = {[1 2 3], [4 5 6], [7 8]};     % 半成品 i 由哪些零配件组成
final_of = [1 2 3];                       % 成品由半成品 1,2,3 组成

% ---------- 绘图 ----------
figure('Position', [80 80 950 720], 'Color', 'w');
hold on; grid on; axis equal off;
view(120, 22);
colormap parula;

% 连线：零配件 → 半成品
for i = 1:3
    for j = semi_of{i}
        p1 = part_pos(j, :); p2 = semi_pos(i, :);
        plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], '-', ...
              'Color', [0.45 0.55 0.65], 'LineWidth', 1.2);
    end
end
% 连线：半成品 → 成品
for i = final_of
    p1 = semi_pos(i, :); p2 = final_pos;
    plot3([p1(1) p2(1)], [p1(2) p2(2)], [p1(3) p2(3)], '-', ...
          'Color', [0.85 0.45 0.30], 'LineWidth', 1.8);
end

% 节点：零配件
scatter3(part_pos(:, 1), part_pos(:, 2), part_pos(:, 3), 260, ...
         'filled', 'MarkerFaceColor', [0.28 0.52 0.78], 'MarkerEdgeColor', 'k');
% 节点：半成品
scatter3(semi_pos(:, 1), semi_pos(:, 2), semi_pos(:, 3), 420, ...
         'filled', 'MarkerFaceColor', [0.36 0.72 0.44], 'MarkerEdgeColor', 'k');
% 节点：成品
scatter3(final_pos(1), final_pos(2), final_pos(3), 640, ...
         'filled', 'MarkerFaceColor', [0.85 0.38 0.28], 'MarkerEdgeColor', 'k');

% 标签（英文避免字体问题）
for i = 1:8
    p = part_pos(i, :);
    text(p(1), p(2) - 0.15, p(3) - 0.25, sprintf('Part %d', i), ...
         'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold', ...
         'Color', [0.15 0.25 0.40]);
end
for i = 1:3
    p = semi_pos(i, :);
    text(p(1), p(2), p(3) + 0.28, sprintf('Semi %d', i), ...
         'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', ...
         'Color', [0.10 0.35 0.18]);
end
text(final_pos(1), final_pos(2), final_pos(3) + 0.38, 'Final Product', ...
     'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', ...
     'Color', [0.55 0.15 0.10]);

title('Assembly Flow: 8 Parts -> 3 Semi-Products -> Final Product', 'FontSize', 13);

% 视角微调与输出
view(-40, 28);
lightangle(-45, 45);
exportgraphics(gcf, fullfile('p3', 'figures', 'fig4_flow_3d.png'), 'Resolution', 300);
fprintf('saved: p3/figures/fig4_flow_3d.png\n');
