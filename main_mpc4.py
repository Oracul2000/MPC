import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- КОНСТАНТЫ ---
m = 1500; Iz = 2500; lf = 1.2; lr = 1.4
Cm = 2000; Crr = 100; Cd = 0.3; Cbf = 2000; Cbr = 1500
Cx = 50000  # Жесткость шин
dt = 0.05

# Регулируемые параметры
max_deviation = 0.2  # Максимальное отклонение от траектории (м)
progress_weight = 10.0  # Вес для продвижения по траектории (больше — быстрее прохождение)

def get_dynamics_model():
    # Состояние: [X, Y, yaw, vx, vy, w, s] — добавили s (параметр траектории)
    x = ca.SX.sym('x', 7)
    # u = [throttle, steering, brakes]
    u = ca.SX.sym('u', 3)
    X, Y, yaw, vx, vy, w, s = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    throttle, steering, brakes = u[0], u[1], u[2]

    # Углы скольжения
    ar = ca.if_else(vx == 0, 0, ca.atan2(vy - lr * w, vx))
    af = ca.if_else(vx == 0, 0, ca.atan2(vy + lf * w, vx) - steering)

    # Продольные силы
    Fdrv = throttle * Cm
    Frrr = Crr * ca.tanh(vx)
    Frrf = Crr * ca.tanh(vx)
    Fdrag = Cd * vx**2
    Fbf = brakes * Cbf * ca.tanh(vx)
    Fbr = brakes * Cbr * ca.tanh(vx)

    # Поперечные силы (линейная модель)
    # Fry = 2 * Cx * ar
    # Ffy = 2 * Cx * af
    D_tire = 1.0 * m * 9.8 / 2  # Максимальная сила на ось (~7350 N)
    Fry = D_tire * ca.tanh(2 * Cx * ar / D_tire)
    Ffy = D_tire * ca.tanh(2 * Cx * af / D_tire)

    # Динамика
    Ftransversal = Fdrv - Frrr - Frrf * ca.cos(steering) - Fdrag - Fbf * ca.cos(steering) - Fbr - Ffy * ca.sin(steering)
    Flateral = -Frrf * ca.sin(steering) - Fbf * ca.sin(steering) + Fry + Ffy * ca.cos(steering)
    Lmoment = -Frrf * ca.sin(steering) * lf - Fbf * ca.sin(steering) * lf - Fry * lr + Ffy * ca.cos(steering) * lf

    # Производные
    dxdt = vx * ca.cos(yaw) - vy * ca.sin(yaw)
    dydt = vx * ca.sin(yaw) + vy * ca.cos(yaw)
    dyawdt = w
    dvxdt = (Ftransversal / m) + vy * w
    dvydt = (Flateral / m) - vx * w
    dwdt = Lmoment / Iz
    dsdt = vx  # Приближение: продвижение по траектории ≈ vx (для малого отклонения)

    xdot = ca.vertcat(dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt, dsdt)
    return ca.Function('f', [x, u], [xdot])

# Задание траектории сплайном (пример: синусоида)
def get_trajectory_spline():
    # Узлы s (от 0 до 10 м, например)
    s_knots = np.linspace(0, 10, 10)
    # Коэффициенты для X(s), Y(s), yaw(s)
    X_coeffs = s_knots  # X = s (прямая по X)
    Y_coeffs = np.sin(s_knots * np.pi / 5) * 2  # Синус по Y
    yaw_coeffs = np.arctan(np.cos(s_knots * np.pi / 5) * (np.pi / 5) * 2)  # Примерный yaw по тангенсу

    # Сплайны
    X_spline = ca.interpolant('X_ref', 'bspline', [s_knots], X_coeffs)
    Y_spline = ca.interpolant('Y_ref', 'bspline', [s_knots], Y_coeffs)
    yaw_spline = ca.interpolant('yaw_ref', 'bspline', [s_knots], yaw_coeffs)

    return X_spline, Y_spline, yaw_spline

if __name__ == "__main__":
    f = get_dynamics_model()
    X_spline, Y_spline, yaw_spline = get_trajectory_spline()

    # --- НАСТРОЙКА MPC ---
    N = 30  # Горизонт (увеличен для траектории)
    dt_mpc = 0.05

    # Символьные переменные
    U = ca.SX.sym('U', 3, N)
    X = ca.SX.sym('X', 7, N + 1)  # Теперь 7 состояний
    P = ca.SX.sym('P', 7)  # Только текущее состояние (цель — сплайн)

    # Веса
    Q = ca.diag([20, 20, 10, 5, 5, 5, 0])  # Нет штрафа на s напрямую
    Q_terminal = 10 * Q
    R = ca.diag([10, 10, 1])
    Q_progress = progress_weight  # Вес для продвижения (на -s для максимизации)

    cost = 0
    g = []  # Равенства (динамика)
    g_dev = []  # Неравенства для отклонения

    # Формирование задачи
    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        s_k = st[6]

        # Reference по сплайну
        X_ref = X_spline(s_k)
        Y_ref = Y_spline(s_k)
        yaw_ref = yaw_spline(s_k)
        ref_state = ca.vertcat(X_ref, Y_ref, yaw_ref, 0, 0, 0, s_k)  # vx_ref=0 для остановки в конце, но можно изменить

        # Ошибка (состояние - ref)
        state_error = st - ref_state
        cost += ca.mtimes([state_error.T, Q, state_error])
        cost += ca.mtimes([con.T, R, con])

        # Cross-track error (CTE) для constraint
        dx = st[0] - X_ref
        dy = st[1] - Y_ref
        theta_ref = yaw_ref
        cte = ca.cos(theta_ref) * dy - ca.sin(theta_ref) * dx  # Проекция на нормаль
        g_dev.append(-max_deviation - cte)  # cte >= -max_dev
        g_dev.append(cte - max_deviation)   # cte <= max_dev

        # Динамика (RK4 для точности)
        st_next = X[:, k + 1]
        k1 = f(st, con)
        k2 = f(st + dt_mpc/2 * k1, con)
        k3 = f(st + dt_mpc/2 * k2, con)
        k4 = f(st + dt_mpc * k3, con)
        st_next_rk4 = st + (dt_mpc/6) * (k1 + 2*k2 + 2*k3 + k4)
        g.append(st_next - st_next_rk4)

    # Терминальный штраф + продвижение (максимизируем s_N)
    state_error_N = X[:, N] - ca.vertcat(X_spline(X[-1, N]), Y_spline(X[-1, N]), yaw_spline(X[-1, N]), 0, 0, 0, X[-1, N])
    cost += ca.mtimes([state_error_N.T, Q_terminal, state_error_N])
    cost -= Q_progress * X[6, N]  # -s_N для максимизации продвижения

    # Оптимизируемые переменные
    OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

    g = ca.vertcat(*g)
    g_dev = ca.vertcat(*g_dev)  # Неравенства для deviation
    nlp_prob = {'f': cost, 'x': OPT_variables, 'g': ca.vertcat(g, g_dev), 'p': P}
    opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 1000,
        'ipopt.tol': 1e-6,
        'print_time': 0
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # --- СИМУЛЯЦИЯ ---
    n_states = 7
    n_controls = 3
    lbx = -ca.inf * np.ones((n_states * (N + 1) + n_controls * N))
    ubx = ca.inf * np.ones((n_states * (N + 1) + n_controls * N))

    # Ограничения на u
    u_start_idx = n_states * (N + 1)
    for i in range(N):
        idx = u_start_idx + i * n_controls
        lbx[idx:idx + 3] = [0, -0.5, 0]
        ubx[idx:idx + 3] = [1, 0.5, 1]

    # Ограничения на g (динамика =0, deviation <=0 для неравенств)
    lbg = np.zeros(n_states * N + 2 * N)  # 0 для равенств, -inf для неравенств? Нет, для g_dev <=0
    ubg = np.zeros(n_states * N)  # Для равенств 0, для g_dev 0 (поскольку g_dev = cte - max <=0 и -cte - max <=0)
    lbg[n_states * N:] = -ca.inf
    ubg = np.concatenate((np.zeros(n_states * N), np.zeros(2 * N)))

    # Начальное состояние (s=0)
    x_curr = np.array([0.0, 0.0, 3.14/2, 0.0, 0.0, 0.0, 0.0])

    # Warm start
    x0_sol = np.zeros((n_states * (N + 1) + n_controls * N))

    # История для визуализации
    X_hist = [x_curr[0]]
    Y_hist = [x_curr[1]]
    vx_hist = [x_curr[3]]
    s_hist = [x_curr[6]]
    time_hist = [0.0]

    print("Start Simulation...")
    for t in range(16):  # Увеличил шаги для траектории
        lbx[0:7] = x_curr
        ubx[0:7] = x_curr

        p_val = x_curr

        sol = solver(x0=x0_sol, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)

        sol_x = sol['x'].full().flatten()
        u_opt_all = sol_x[u_start_idx:]
        u_apply = u_opt_all[0:3]

        x_next_val = f(x_curr, u_apply).full().flatten() * dt + x_curr

        x_curr = x_next_val

        x0_sol = sol_x

        X_hist.append(x_curr[0])
        Y_hist.append(x_curr[1])
        vx_hist.append(x_curr[3])
        s_hist.append(x_curr[6])
        time_hist.append(time_hist[-1] + dt)

        print(f"Step {t}: X={x_curr[0]:.2f}, Y={x_curr[1]:.2f}, s={x_curr[6]:.2f}, Vx={x_curr[3]:.2f}, Throttle={u_apply[0]:.3f}, Steer={u_apply[1]:.3f}, Brakes={u_apply[2]:.3f}")

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Траектория
    s_ref = np.linspace(0, 10, 100)
    X_ref_plot = [X_spline(si).full()[0][0] for si in s_ref]
    Y_ref_plot = [Y_spline(si).full()[0][0] for si in s_ref]
    axs[0].plot(X_ref_plot, Y_ref_plot, 'r--', label='Траектория (сплайн)')
    axs[0].plot(X_hist, Y_hist, 'b-', label='Фактическая траектория')
    axs[0].plot(X_hist[0], Y_hist[0], 'go', label='Начало')
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].legend()
    axs[0].grid(True)

    # Продольная скорость
    axs[1].plot(time_hist, vx_hist, 'g-', label='Продольная скорость (vx)')
    axs[1].set_xlabel('Время (с)')
    axs[1].set_ylabel('Скорость (м/с)')
    axs[1].legend()
    axs[1].grid(True)

    # Прогресс по s
    axs[2].plot(time_hist, s_hist, 'm-', label='Прогресс по траектории (s)')
    axs[2].set_xlabel('Время (с)')
    axs[2].set_ylabel('s')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    # plt.show()

    # Сохранение графика в файл
    plt.savefig('trajectory_plot.png')