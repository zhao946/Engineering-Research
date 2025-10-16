function predictions = recursive_prediction(net, last_window, num_steps)
    lookback = length(last_window);
    predictions = zeros(num_steps, 1);
    % 初始化
    current_window = last_window(:)'; 

     % 填充
    for i = 1:min(lookback, length(last_window))
        predictions(i) = last_window(i);
    end

    % 开始预测
    start_idx = lookback + 1;
    
    for i = start_idx:num_steps
        % 预测下一个点
        X_input = cell(1, 1);
        X_input{1} = reshape(current_window, 1, lookback);
        next_pred = predict(net, X_input);

        % 存储
        predictions(i) = next_pred;
        
        % 更新
        current_window = [current_window(2:end), next_pred];

        % 显示
        if mod(i, 100) == 0
            fprintf('递归预测进度: %d/%d\n', i, num_steps);
        end
    end
    fprintf('递归预测完成。残差范围: [%.6f, %.6f] m\n', min(predictions), max(predictions));
   
end