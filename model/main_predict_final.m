% =========================================================================
% ===== 主程序=====
% =========================================================================

%% ===== 阶段一=====
clear; clc; close all;
fprintf('阶段一: 初始化与数据加载...\n');

% --- 定义冲击角度 (新添加) ---
impact_angle_deg = 90; % <--- 冲击角度
fprintf('-> 当前冲击角度设置为: %d 度\n', impact_angle_deg);

params_phys_known = [18.01, 27.8e3, 1.25];     % [JIC (J/m^2), mu (Pa), alpha]

% 将角度从度转换为弧度，因为MATLAB的sin()函数使用弧度
impact_angle_rad = deg2rad(impact_angle_deg); 
params_jet_known = [1e6, 44.72, 0.2e-3, 1000, impact_angle_rad]; % [P (Pa), v (m/s), D (m), rho (kg/m^3), angle_rad]

try
    train_data = readtable('table1.xls');
    T_known_si = train_data{:,1} * 1e-3; % s
    H_known_si = train_data{:,2} * 1e-3; % m
catch
    error('无法读取训练数据 "table1.xls"。请确保文件存在且格式正确。');
end

% --- 加载10ms验证数据 (用于最终绘图对比)
try
    validation_data = readtable('table2.xls');
    T_validation_ms = validation_data{:,1};
    H_validation_mm = validation_data{:,2};
catch
    warning('无法读取验证数据 "table2.xls"。最终绘图将缺少真实数据对比。');
    T_validation_ms = []; H_validation_mm = [];
end


%% ===== 阶段二 =====
fprintf('阶段二: GWO 优化开始 (这可能需要一些时间)...\n');

% --- GWO 参数
SearchAgents_no = 50; % 狼群数量
Max_iter = 150;      % 最大迭代次数

% --- 待优化参数的维度和边界 [eta, beta, n, hidden_units, learn_rate]
dim = 5;
lb = [0.1, 0.01, 0.2, 20,  0.001]; % 下限
ub = [6.0, 1.0, 1.0, 150, 0.01];  % 上限

% --- 创建适应度函数的句柄
fobj = @(params) fitnessFunction_final(params, T_known_si, H_known_si, params_phys_known, params_jet_known);

% --- 执行
[Best_score, Best_pos, GWO_cg_curve] = GWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj);

fprintf('GWO优化完成!\n');
fprintf(' -> 最优适应度 (RMSE): %e\n', Best_score);
fprintf(' -> 最优物理参数 [eta, beta, n]: [%f, %f, %f]\n', Best_pos(1), Best_pos(2), Best_pos(3));
fprintf(' -> 最优LSTM参数 [hidden_units, learn_rate]: [%d, %f]\n', round(Best_pos(4)), Best_pos(5));


%% ===== 阶段三=====
fprintf('阶段三: 使用最优参数进行最终预测...\n');

% --- 提取参数
best_phys_params = Best_pos(1:3);
best_lstm_params = [round(Best_pos(4)), Best_pos(5)];

% --- 步骤3.1
T_future = linspace(0, 10e-3, 500)'; % 预测10ms, 500个点
H_phys_future = physics_model_damped(T_future, best_phys_params, params_phys_known, params_jet_known);

% --- 步骤3.2
% 1. 计算0.01ms内的最优残差
H_phys_known = physics_model_damped(T_known_si, best_phys_params, params_phys_known, params_jet_known);
Residual_known = H_known_si - H_phys_known;

% 2. 准备数据
lookback = 5; % 定义回看窗口大小
[X_train, Y_train] = create_lstm_samples(Residual_known, lookback);

% 3. 训练
final_net = train_residual_lstm(X_train, Y_train, best_lstm_params);

% --- 步骤3.3
num_prediction_steps = length(T_future);
last_known_residual_window = Residual_known(end-lookback+1:end);
Residual_future = recursive_prediction(final_net, last_known_residual_window, num_prediction_steps);

% --- 步骤3.4
H_final_future = H_phys_future + Residual_future;

%% ===== 阶段 3.5
fprintf('阶段 3.5: 计算模型在0.01ms训练数据上的拟合结果...\n');

% --- 步骤 3.5.1
H_phys_train = physics_model_damped(T_known_si, best_phys_params, params_phys_known, params_jet_known);

% --- 步骤 3.5.2
Residual_lstm_train_pred = predict(final_net, X_train);

% --- 步骤 3.5.3
Residual_correction_train = zeros(size(H_known_si));
% LSTM的预测是从第 lookback+1 个点开始的，所以将预测值填充到对应位置
Residual_correction_train(lookback+1:end) = Residual_lstm_train_pred;

% 拟合结果
H_hybrid_train = H_phys_train + Residual_correction_train;

% 结果转换
H_hybrid_train_mm = H_hybrid_train * 1000;

fprintf(' -> 训练数据拟合结果计算完成!\n');

%% ===== 阶段四 =====
fprintf('\n阶段四: 生成详细可视化结果...\n');

% 1. 主要预测结果图（多子图展示）
figure('Name', '混合模型预测结果详细分析', 'Position', [50, 60, 1200, 700]);

% --- 子图1
subplot(2, 3, 1);
hold on;
if ~isempty(T_validation_ms)
    plot(T_validation_ms, H_validation_mm, 'r-', 'LineWidth', 2.5, 'DisplayName', '真实数据');
end
plot(T_known_si*1000, H_known_si*1000, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 6, 'DisplayName', '训练数据');
plot(T_future*1000, H_phys_future*1000, 'b--', 'LineWidth', 2, 'DisplayName', '物理模型');
plot(T_future*1000, H_final_future*1000, 'g-', 'LineWidth', 2.5, 'DisplayName', '混合模型');
xlabel('时间 (ms)'); ylabel('深度 (mm)');
title('(a) 完整预测对比');
legend('Location', 'southeast');
grid on; box on;

% --- 子图3：残差分析
subplot(2, 3, 3);
hold on;
% 计算并显示残差
residual_phys = abs(interp1(T_future, H_phys_future, T_validation_ms/1000)*1000 - H_validation_mm);
residual_hybrid = abs(interp1(T_future, H_final_future, T_validation_ms/1000)*1000 - H_validation_mm);
plot(T_validation_ms, residual_phys, 'b--', 'LineWidth', 1.5, 'DisplayName', '物理模型残差');
plot(T_validation_ms, residual_hybrid, 'g-', 'LineWidth', 2, 'DisplayName', '混合模型残差');
xlabel('时间 (ms)'); ylabel('残差 (mm)');
title('(c) 预测残差对比');
legend('Location', 'best');
grid on; box on;

% 添加数值标签
for i = 1:size(metrics_data, 1)
    for j = 1:size(metrics_data, 2)
        text(i + (j-1.5)*0.15, metrics_data(i,j), sprintf('%.3f', metrics_data(i,j)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
end

% 2. GWO
figure('Name', 'GWO优化过程详细分析', 'Position', [100, 100, 1200, 600]);

% --- 子图1：收敛曲线
subplot(1, 3, 1);
semilogy(1:Max_iter, GWO_cg_curve, 'm-', 'LineWidth', 2);
xlabel('迭代次数'); ylabel('最优适应度 (RMSE)');
title('(a) GWO收敛曲线 (对数尺度)');
grid on; box on;

% 3. 训练数据拟合质量评估
figure('Name', '训练数据拟合评估', 'Position', [150, 150, 1000, 600]);

% --- 子图1：训练数据拟合
subplot(2, 2, 1);
plot(T_known_si*1000, H_known_si*1000, 'ko', 'MarkerSize', 8, 'DisplayName', '训练数据');
hold on;
H_phys_train = physics_model_damped(T_known_si, best_phys_params, params_phys_known, params_jet_known);
plot(T_known_si*1000, H_phys_train*1000, 'b-', 'LineWidth', 2, 'DisplayName', '物理模型拟合');
xlabel('时间 (ms)'); ylabel('深度 (mm)');
title('(a) 训练数据物理模型拟合');
legend('Location', 'best');
grid on; box on;

% --- 子图2：残差分布
subplot(2, 2, 2);
train_residuals = (H_known_si - H_phys_train) * 1000;  % mm
histogram(train_residuals, 20, 'FaceColor', 'c');
xlabel('残差 (mm)'); ylabel('频数');
title('(b) 训练残差分布');
grid on; box on;

% 4. 性能总结表
fprintf('\n========== 性能评估总结 ==========\n');
fprintf('物理模型:\n');
fprintf('  RMSE: %.4f mm\n', RMSE_phys);
fprintf('  MAE:  %.4f mm\n', MAE_phys);
fprintf('  R²:   %.4f\n', R2_phys);
fprintf('\n混合模型:\n');
fprintf('  RMSE: %.4f mm (改进: %.1f%%)\n', RMSE_hybrid, (RMSE_phys-RMSE_hybrid)/RMSE_phys*100);
fprintf('  MAE:  %.4f mm (改进: %.1f%%)\n', MAE_hybrid, (MAE_phys-MAE_hybrid)/MAE_phys*100);
fprintf('  R²:   %.4f (改进: %.1f%%)\n', R2_hybrid, (R2_hybrid-R2_phys)/(1-R2_phys)*100);
fprintf('\n最优物理参数:\n');
fprintf('  η = %.4f\n', best_phys_params(1));
fprintf('  β = %.4f\n', best_phys_params(2));
fprintf('  n = %.4f\n', best_phys_params(3));
fprintf('======================================\n');

fprintf('\n所有分析完成！\n');