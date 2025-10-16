% =========================================================================
% =====Fitness Function=====
% =========================================================================
function rmse = fitnessFunction_final(params, T_known, H_known, params_phys_known, params_jet_known)
    % 1. 参数
    phys_params_unknown = params(1:3); % [eta, beta, n]
    lstm_params = [round(params(4)), params(5)]; % [hidden_units, learn_rate]

    % 2. 预测
    H_phys_pred = physics_model_damped(T_known, phys_params_unknown, params_phys_known, params_jet_known);
    
    % 3. 残差
    Residual_known = H_known - H_phys_pred;
    
    % 4. 数据
    lookback = 5;
    [X_train, Y_train] = create_lstm_samples(Residual_known, lookback);
    
    % 5. 训练
    temp_net = train_residual_lstm(X_train, Y_train, lstm_params);

    % 6. 计算
    Residual_pred = predict(temp_net, X_train);
    
    H_hybrid_pred = H_phys_pred(lookback+1:end) + Residual_pred;
    
    % 7. 返回
    rmse = sqrt(mean((H_hybrid_pred - H_known(lookback+1:end)).^2));
    
    if isnan(rmse) || isinf(rmse)
        rmse = 1e10; % 返回一个很大的惩罚值
    end
end