% =========================================================================
% =====model=====
% =========================================================================
% =========================================================================

function H_phys = physics_model_damped(t_span, params_unknown, params_phys_known, params_jet_known)
    % 解码所有传入的参数
    
    % 未知参数 (由GWO优化)
    eta = params_unknown(1);    
    beta = params_unknown(2);   
    n = params_unknown(3);      

    % 物理参数
    JIC = params_phys_known(1); 
    mu = params_phys_known(2);  
    alpha = params_phys_known(3); 

    % 水射流参数
    P_jet = params_jet_known(1); 
    v_jet = params_jet_known(2); 
    D_jet = params_jet_known(3); 
    rho_water = params_jet_known(4); 

    % --- 冲击角度 ---
    impact_angle_rad = params_jet_known(5); 
    % --- 1. 最小穿透压力 Pw_min ---
    % 找到材料抵抗射流开始穿透所需的最小压力
    % 通过最小化一个能量函数来实现，结合了断裂能和塑性变形能。
    f_pw = @(d_ratio) calculate_Pw(d_ratio, D_jet, JIC, mu, alpha);
    options = optimset('TolX', 1e-9, 'Display', 'off');
    [~, Pw_min] = fminbnd(f_pw, 0.001, 0.999, options);
    
    % --- 2. 定义---
    odefun = @(t, h) define_damped_ode(t, h, Pw_min, eta, n, beta, P_jet, D_jet, mu, impact_angle_rad);
    
    % --- 3. 求解 ---
    ode_options = odeset('NonNegative', 1); % 确保深度h不会是负数
    [~, H_phys] = ode45(odefun, t_span, 0, ode_options); % 初始条件: h(0) = 0
end


% =========================================================================
% =====函数=====
% =========================================================================

%% --- 函数1
function Pw = calculate_Pw(d_ratio, D, JIC, mu, alpha)
    % 在给定条件下，产生穿透所需的压力。
    % 由两部分组成：
    % 克服材料断裂韧性所需的能量 。
    % 引起材料塑性剪切变形所需的能量。
    
    % Pw = (2 * JIC / D) * (1 / d_ratio) + mu * (1 - d_ratio)^alpha;
    
    fracture_term = (2 * JIC / D) * (1 / d_ratio);
    plastic_term = mu * (1 - d_ratio)^alpha;
    
    Pw = fracture_term + plastic_term;
end


%% --- 函数2 ---
function dhdt = define_damped_ode(t, h, Pw_min, eta, n, beta, P_jet, D_jet, mu, impact_angle_rad)
    % 计算任意时刻t和深度h下的深度变化率 dh/dt。
    % dh/dt = 驱动项 - 阻尼项

    % --- A. 计算驱动项 F_driving(h) ---
    % F_driving(h) = K_drive * G(h, eta, D_jet)
    
    % A.1. 计算驱动常数 K_drive
    P_jet_normal = P_jet * (sin(impact_angle_rad)^2);
    
    % 计算有效压力
    effective_pressure = P_jet_normal - Pw_min;
    
    C_n = ((2 - n)^2) / (2 * n);
    if effective_pressure <= 0
        K_drive = 0;
    else
        K_drive = C_n * (effective_pressure / (2 * mu));
    end

    % A.2. 计算几何反馈函数 G(h, eta, D_jet)
    
    % 切割前缘的有效直径随着深度h而扩大
    % D_effective = D_jet + 2 * h / eta;
    % G = (D_jet / D_effective)^2;
    
    geometric_feedback = (1 + (2 * h) / (eta * D_jet))^(-2);
    
    % 驱动项
    driving_term = K_drive * geometric_feedback;

    % --- B. 阻尼项 F_damping(h) ---
    damping_term = beta * h;
    
    % --- C. 计算 ---
    dhdt = driving_term - damping_term;
    
    % 物理约束
    if h > 0 && dhdt < 0
        dhdt = 0;
    end
end