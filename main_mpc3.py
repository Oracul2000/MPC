import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# --- КОНСТАНТЫ (Заглушки, так как файл не приложен) ---
m = 1500; Iz = 2500; lf = 1.2; lr = 1.4
Cm = 2000; Crr = 100; Cd = 0.3; Cbf = 2000; Cbr = 1500
Cx = 50000  # Жесткость шин
dt = 0.05

def get_dynamics_model():
    # [X, Y, yaw, vx, vy, w]
    x = ca.SX.sym('x', 6)
    # u = [throttle, steering_angle, brakes]
    u = ca.SX.sym('u', 3)
    X, Y, yaw, vx, vy, w = x[0], x[1], x[2], x[3], x[4], x[5]
    throttle, steering, brakes = u[0], u[1], u[2]

    # Углы скольжения (Slip angles)
    ar = ca.if_else(vx == 0, 0, ca.atan2(vy - lr * w, vx))
    af = ca.if_else(vx == 0, 0, ca.atan2(vy + lf * w, vx) - steering)

    # Продольные силы
    Fdrv = throttle * Cm
    Frrr = Crr * ca.tanh(vx)
    Frrf = Crr * ca.tanh(vx)
    Fdrag = Cd * vx**2
    Fbf = brakes * Cbf * ca.tanh(vx)
    Fbr = brakes * Cbr * ca.tanh(vx)

    # Поперечные силы (Линейная модель, как в C++)
    Fry = 2 * Cx * ar  # Умножение на 2, как в C++ (по шинам?)
    Ffy = 2 * Cx * af

    # Полная динамика (выровнено с C++)
    Ftransversal = Fdrv - Frrr - Frrf * ca.cos(steering) - Fdrag - Fbf * ca.cos(steering) - Fbr - Ffy * ca.sin(steering)
    Flateral = -Frrf * ca.sin(steering) - Fbf * ca.sin(steering) + Fry + Ffy * ca.cos(steering)
    Lmoment = -Frrf * ca.sin(steering) * lf - Fbf * ca.sin(steering) * lf - Fry * lr + Ffy * ca.cos(steering) * lf

    # Производные состояния
    dxdt = vx * ca.cos(yaw) - vy * ca.sin(yaw)
    dydt = vx * ca.sin(yaw) + vy * ca.cos(yaw)
    dyawdt = w
    dvxdt = (Ftransversal / m) + vy * w
    dvydt = (Flateral / m) - vx * w
    dwdt = Lmoment / Iz
    xdot = ca.vertcat(dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt)

    return ca.Function('f', [x, u], [xdot])

if __name__ == "__main__":
    f = get_dynamics_model()

    # --- НАСТРОЙКА MPC ---
    N = 100  # Горизонт (разумный размер для скорости)
    dt_mpc = 0.05  # Совпадает с dt симуляции

    # Символьные переменные для всего горизонта
    U = ca.SX.sym('U', 3, N)  # Управление на N шагов
    X = ca.SX.sym('X', 6, N + 1)  # Состояние на N+1 шагов
    P = ca.SX.sym('P', 6 + 6)  # Параметры: [x_current, x_target]

    # Веса (Матрицы Q и R)
    Q = ca.diag([20, 20, 10, 5, 5, 5])  # Штрафуем все состояния
    Q_terminal = 10 * Q  # Терминальный штраф для фокуса на цели
    R = ca.diag([10, 10, 1])  # Штрафуем резкое управление

    cost = 0
    g = []  # Ограничения равенства (динамика)

    # Целевое состояние из параметров P
    current_state_ref = P[:6]
    target_state_ref = P[6:]

    # Формирование задачи
    for k in range(N):
        st = X[:, k]
        con = U[:, k]

        # Целевая функция (квадратичная ошибка)
        state_error = st - target_state_ref
        cost += ca.mtimes([state_error.T, Q, state_error])
        cost += ca.mtimes([con.T, R, con])

        # Динамика (Multiple Shooting)
        st_next = X[:, k + 1]
        k1 = f(st, con)
        st_next_euler = st + dt_mpc * k1
        g.append(st_next - st_next_euler)

    # Добавляем терминальный штраф
    state_error_N = X[:, N] - target_state_ref
    cost += ca.mtimes([state_error_N.T, Q_terminal, state_error_N])

    # Вектор оптимизируемых переменных
    OPT_variables = ca.vertcat(
        ca.reshape(X, -1, 1),  # Сначала состояния 6*(N+1)
        ca.reshape(U, -1, 1)  # Потом управление 3*N
    )

    g = ca.vertcat(*g)
    nlp_prob = {
        'f': cost,
        'x': OPT_variables,
        'g': g,
        'p': P
    }
    opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 500,  # Увеличено для лучшей сходимости
        'ipopt.acceptable_tol': 1e-6,
        'ipopt.acceptable_obj_change_tol': 1e-4,
        'print_time': 0
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # --- СИМУЛЯЦИЯ ---

    # Ограничения переменных
    n_states = 6
    n_controls = 3

    lbx = -ca.inf * np.ones((n_states * (N + 1) + n_controls * N))
    ubx = ca.inf * np.ones((n_states * (N + 1) + n_controls * N))

    # Ограничения на управление (throttle, steering, brakes)
    u_start_idx = n_states * (N + 1)

    for i in range(N):
        idx = u_start_idx + i * n_controls
        # Throttle [0, 1], Steering [-0.5, 0.5], Brakes [0, 1]
        lbx[idx:idx + 3] = [0, -0.5, 0]
        ubx[idx:idx + 3] = [1, 0.5, 1]

    # Ограничения равенства g (динамика) должны быть 0
    lbg = np.zeros(n_states * N)
    ubg = np.zeros(n_states * N)

    # Начальное состояние
    x_curr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Начинаем с покоя

    # Цель: сместиться на (1,1), сохраняя нулевые скорости
    x_ref_val = np.array([10.0, 10.0, 0.0, 0.0, 0.0, 0.0])

    # Warm start (начальное приближение решения)
    x0_sol = np.zeros((n_states * (N + 1) + n_controls * N))

    # Сбор истории для визуализации
    X_hist = [x_curr[0]]
    Y_hist = [x_curr[1]]
    vx_hist = [x_curr[3]]
    time_hist = [0.0]

    print("Start Simulation...")
    for t in range(120):
        # 1. ОБНОВЛЯЕМ ОГРАНИЧЕНИЯ ДЛЯ НАЧАЛЬНОГО СОСТОЯНИЯ
        lbx[0:6] = x_curr
        ubx[0:6] = x_curr

        # 2. Параметры (текущее + цель)
        p_val = np.concatenate((x_curr, x_ref_val))

        # 3. Решаем
        sol = solver(x0=x0_sol, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)

        # 4. Извлекаем управление
        sol_x = sol['x'].full().flatten()
        u_opt_all = sol_x[u_start_idx:]
        u_apply = u_opt_all[0:3]  # Первое управление

        # 5. Применяем к модели (симуляция реальности)
        x_next_val = f(x_curr, u_apply).full().flatten() * dt + x_curr

        # Обновляем текущее состояние
        x_curr = x_next_val

        # Warm start для следующего шага
        x0_sol = sol_x

        # Сбор данных
        X_hist.append(x_curr[0])
        Y_hist.append(x_curr[1])
        vx_hist.append(x_curr[3])
        time_hist.append(time_hist[-1] + dt)

        # Лог
        print(f"Step {t}: X={x_curr[0]:.2f}, Y={x_curr[1]:.2f}, Vx={x_curr[3]:.2f}, Throttle={u_apply[0]:.3f}, Steer={u_apply[1]:.3f}, Brakes={u_apply[2]:.3f}")

    # --- ВИЗУАЛИЗАЦИЯ ---
    # vx - это продольная скорость относительно автомобиля (в локальной системе координат машины),
    # т.е. скорость вперед/назад. Это не глобальная скорость по оси X, а именно вдоль оси автомобиля.

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Траектория движения
    axs[0].plot(X_hist, Y_hist, 'b-', label='Траектория')
    axs[0].plot(X_hist[0], Y_hist[0], 'go', label='Начало (0,0)')
    axs[0].plot(X_hist[-1], Y_hist[-1], 'ro', label='Конец')
    axs[0].plot(x_ref_val[0], x_ref_val[1], 'rx', label='Цель (1,1)', markersize=10)
    axs[0].set_xlabel('X')
    axs[0].set_ylabel('Y')
    axs[0].set_title('Траектория машины')
    axs[0].legend()
    axs[0].grid(True)

    # 2. График продольной скорости (vx)
    axs[1].plot(time_hist, vx_hist, 'g-', label='Продольная скорость (vx)')
    axs[1].set_xlabel('Время (с)')
    axs[1].set_ylabel('Скорость (м/с)')
    axs[1].set_title('Продольная скорость')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    # plt.show()
    plt.savefig('./trajectory_and_speed.png')