% =========================================================================
% =====Grey Wolf Optimizer=====
% =========================================================================
function [Alpha_score, Alpha_pos, Convergence_curve] = GWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj)
    % 初始化
    Alpha_pos = zeros(1, dim);
    Alpha_score = inf; 
    Beta_pos = zeros(1, dim);
    Beta_score = inf;
    Delta_pos = zeros(1, dim);
    Delta_score = inf;

    % 初始化
    Positions = initialization(SearchAgents_no, dim, ub, lb);
    Convergence_curve = zeros(1, Max_iter);

    % 主循环
    for l = 1:Max_iter
        for i = 1:size(Positions, 1)
            % 边界检查
            Flag4ub = Positions(i, :) > ub;
            Flag4lb = Positions(i, :) < lb;
            Positions(i, :) = (Positions(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            
            % 适应度
            fitness = fobj(Positions(i, :));
            
            % 更新
            if fitness < Alpha_score
                Alpha_score = fitness;
                Alpha_pos = Positions(i, :);
            end
            if fitness > Alpha_score && fitness < Beta_score
                Beta_score = fitness;
                Beta_pos = Positions(i, :);
            end
            if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
                Delta_score = fitness;
                Delta_pos = Positions(i, :);
            end
        end
        
        a = 2 - l * ((2) / Max_iter); % a 从 2 线性递减到 0
        
        % 更新
        for i = 1:size(Positions, 1)
            for j = 1:size(Positions, 2)
                r1 = rand(); 
                r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - A1 * D_alpha;
                
                r1 = rand();
                r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - A2 * D_beta;
                
                r1 = rand();
                r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                X3 = Delta_pos(j) - A3 * D_delta;
                
                Positions(i, j) = (X1 + X2 + X3) / 3;
            end
        end
        Convergence_curve(l) = Alpha_score;
        fprintf('迭代 %d: 最优RMSE = %e\n', l, Alpha_score);
    end
end

function X = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        X = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            X(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end