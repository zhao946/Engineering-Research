function [X, Y] = create_lstm_samples(data, lookback)
    % 创建样本
    n = length(data);
    n_samples = n - lookback;
    X = cell(n_samples, 1);
    Y = zeros(n_samples, 1);
    for i = 1:n_samples
        % 关键：每个样本是 1 × lookback 格式
        X = reshape(data(i:i+lookback-1), 1, lookback);
        Y = data(i+lookback);
    end
    % cell数组格式，每个cell是一个序列
    X = num2cell(X, 2);
end