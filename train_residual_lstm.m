function net = train_residual_lstm(X_train, Y_train, lstm_params)
    hidden_units = lstm_params(1);
    learn_rate = lstm_params(2);
    
    layers = [ ...
        sequenceInputLayer(1)
        lstmLayer(hidden_units, 'OutputMode', 'last')
        fullyConnectedLayer(1)
        regressionLayer];
    
    options = trainingOptions('adam', ...
        'MaxEpochs', 50, ...
        'GradientThreshold', 1, ...
        'InitialLearnRate', learn_rate, ...
        'Verbose', false);
        
    net = trainNetwork(X_train, Y_train, layers, options);
end