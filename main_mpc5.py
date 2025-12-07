import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Попытка импорта, чтобы код не падал, если модуля нет при проверке
try:
    from cpp_port import car_model 
except ImportError:
    print("Внимание: Модуль cpp_port не найден. Симуляция не запустится.")
    car_model = None

# --- КОНСТАНТЫ ---
# Физика
m = 300.0
lf = 0.721
lr = 0.823
Cm = 3600.0
Crr = 200.0
Cd = 1.53
Cbf = 5411.0
Cbr = 2650.0
Cx = -20000.0
Iz = 134.0
Iw = 0.25
b = 0.669

# Настройки времени
dt_sim = 0.001        # Шаг симуляции физики (100 Hz)
dt_mpc = 0.05        # Шаг предсказания MPC (20 Hz)
mpc_steps = int(dt_mpc / dt_sim) # Сколько шагов симуляции держать одно управление (5)

# Параметры MPC
N = 30               # Горизонт прогноза
max_deviation = 1.0  # Максимальное отклонение от траектории
progress_weight = 10.0 # Вес прогресса (аккуратнее с ним, чтобы не нарушать физику)
MAX_STEER = 0.6      # Максимальный угол поворота колес (радианы, ~35 градусов)

def get_dynamics_model():
    """
    Создает CasADi функцию динамики автомобиля.
    """
    x = ca.SX.sym('x', 7)  # [X, Y, yaw, vx, vy, w, s]
    u = ca.SX.sym('u', 3)  # [throttle, steering, brakes]

    X, Y, yaw, vx, vy, w, s = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    throttle, steering, brakes = u[0], u[1], u[2]

    # Углы скольжения
    # Добавляем малую константу 1e-5 к знаменателю, чтобы избежать деления на 0 при старте
    ar = ca.if_else(vx < 0.1, 0, ca.atan2(vy - lr * w, vx + 1e-5))
    af = ca.if_else(vx < 0.1, 0, ca.atan2(vy + lf * w, vx + 1e-5) - steering)

    # Силы
    Fdrv = throttle * Cm
    Frrr = Crr * ca.tanh(vx)
    Frrf = Crr * ca.tanh(vx) # Упрощено, можно добавить учет нагрузки
    Fdrag = Cd * vx**2
    Fbf = brakes * Cbf * ca.tanh(vx)
    Fbr = brakes * Cbr * ca.tanh(vx)

    # Pacejka / Simplified Tire Model
    D_tire = 1.0 * m * 9.81 / 2
    Fry = D_tire * ca.tanh(2 * Cx * ar / D_tire)
    Ffy = D_tire * ca.tanh(2 * Cx * af / D_tire)

    # Уравнения движения
    Ftransversal = Fdrv - Frrr - Frrf * ca.cos(steering) - Fdrag - Fbf * ca.cos(steering) - Fbr - Ffy * ca.sin(steering)
    Flateral = -Frrf * ca.sin(steering) - Fbf * ca.sin(steering) + Fry + Ffy * ca.cos(steering)
    Lmoment = -Frrf * ca.sin(steering) * lf - Fbf * ca.sin(steering) * lf - Fry * lr + Ffy * ca.cos(steering) * lf

    dxdt = vx * ca.cos(yaw) - vy * ca.sin(yaw)
    dydt = vx * ca.sin(yaw) + vy * ca.cos(yaw)
    dyawdt = w
    dvxdt = (Ftransversal / m) + vy * w
    dvydt = (Flateral / m) - vx * w
    dwdt = Lmoment / Iz
    
    # Проекция скорости на путь (упрощенная)
    dsdt = vx # Можно улучшить до: (vx * ca.cos(epsi) - vy * ca.sin(epsi)) / (1 - ey * curvature)

    xdot = ca.vertcat(dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt, dsdt)

    return ca.Function('f', [x, u], [xdot])

def gen_double_P_track(x1=5, y1=25, r1=5):
    """Генерация точек трассы."""
    first_sector = {'x': np.zeros(y1 - r1), 'y': np.linspace(0, y1 - r1, y1 - r1)}
    sector2points_qty = 50 * r1
    theta2sector = np.pi - np.linspace(2 * np.pi / sector2points_qty, 3 * np.pi / 2 - 2 * np.pi / sector2points_qty, sector2points_qty - 1)
    second_sector = {'x': r1 * np.cos(theta2sector) + r1, 'y': r1 * np.sin(theta2sector) + y1 - r1}
    third_sector = {'x': np.ones(y1 - 2 * r1) * r1, 'y': np.linspace(0, y1 - 2 * r1, y1 - 2 * r1)}
    
    x_merged = np.concatenate((first_sector['x'], second_sector['x'], third_sector['x']))
    y_merged = np.concatenate((first_sector['y'], second_sector['y'], third_sector['y'][::-1]))
    
    x_merged = np.concatenate((x_merged, x_merged + r1))
    y_merged = np.concatenate((y_merged, -y_merged))
    
    points = np.stack((x_merged, y_merged), axis=1)
    return points, x_merged, y_merged

def get_trajectory_spline(path):
    """
    Строит сплайн траектории и возвращает ОЧИЩЕННЫЕ данные для синхронизации.
    """
    # 1. Очистка от дубликатов
    diffs = np.diff(path, axis=0)
    lengths_temp = np.sqrt(np.sum(diffs**2, axis=1))
    mask = lengths_temp > 1e-2 # Порог чуть выше машинной точности
    cleaned_path = np.concatenate((path[0:1], path[1:][mask]))
    
    # 2. Пересчет длин дуг (s) для очищенного пути
    diffs = np.diff(cleaned_path, axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    s_knots = np.cumsum(np.concatenate(([0], lengths)))
    
    # 3. Расчет угла с unwrap (ВАЖНО для замкнутых трасс)
    dx = np.gradient(cleaned_path[:, 0], s_knots)
    dy = np.gradient(cleaned_path[:, 1], s_knots)
    yaw_raw = np.arctan2(dy, dx)
    yaw_unwrapped = np.unwrap(yaw_raw) # Убирает скачки pi -> -pi

    # 4. Создание сплайнов
    X_spline = ca.interpolant('X_ref', 'bspline', [s_knots], cleaned_path[:, 0])
    Y_spline = ca.interpolant('Y_ref', 'bspline', [s_knots], cleaned_path[:, 1])
    yaw_spline = ca.interpolant('yaw_ref', 'bspline', [s_knots], yaw_unwrapped)
    
    # Возвращаем сплайны + данные, на которых они построены
    return X_spline, Y_spline, yaw_spline, s_knots[-1], cleaned_path, s_knots

def find_closest_s(x, y, cleaned_path, s_knots):
    """
    Находит s, используя те же данные, что и сплайн.
    """
    dists = (cleaned_path[:,0] - x)**2 + (cleaned_path[:,1] - y)**2
    idx = np.argmin(dists)
    return s_knots[idx]

if __name__ == "__main__":
    # 1. Подготовка модели и трассы
    f = get_dynamics_model()
    raw_path, center_line_x, center_line_y = gen_double_P_track()
    
    # Получаем сплайны и, что важно, ОЧИЩЕННЫЙ путь
    X_spline, Y_spline, yaw_spline, max_s_track, clean_path, s_knots_ref = get_trajectory_spline(raw_path)

    # 2. Формулировка MPC задачи
    U = ca.SX.sym('U', 3, N)       # [throttle, steering, brakes]
    X = ca.SX.sym('X', 7, N + 1)   # State
    P = ca.SX.sym('P', 7)          # Initial state parameter

    # Веса
    Q = ca.diag([10, 10, 50, 0.1, 0, 0, 0]) # X, Y, Yaw, vx, vy, w, s
    Q_terminal = 10 * Q
    R = ca.diag([1, 10, 50])            # Штраф за управление

    cost = 0
    g = []     # Ограничения динамики
    g_dev = [] # Ограничения трассы

    st = X[:, 0]
    g.append(st - P) # Начальное условие

    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        
        # Референс
        s_k = st[6]
        X_ref = X_spline(s_k)
        Y_ref = Y_spline(s_k)
        yaw_ref = yaw_spline(s_k)
        vx_ref = 10.0 # Целевая скорость
        
        # Вектор ошибки (x, y, yaw, vx, vy, w, s)
        # Обратите внимание: сравниваем yaw напрямую, так как оба (авто и реф) будут "развернуты" (unwrapped)
        ref_state = ca.vertcat(X_ref, Y_ref, yaw_ref, vx_ref, 0, 0, s_k)
        state_error = st - ref_state
        
        # Cost Function
        cost += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([con.T, R, con])
        
        # Расчет отклонения (CTE) для ограничений
        dx_err = st[0] - X_ref
        dy_err = st[1] - Y_ref
        # Проекция ошибки на нормаль к трассе
        cte = -ca.sin(yaw_ref) * dx_err + ca.cos(yaw_ref) * dy_err
        
        g_dev.append(cte) # Добавляем в список ограничений (границы зададим в lbg/ubg)

        # Динамика (RK4)
        st_next = X[:, k + 1]
        k1 = f(st, con)
        k2 = f(st + dt_mpc/2 * k1, con)
        k3 = f(st + dt_mpc/2 * k2, con)
        k4 = f(st + dt_mpc * k3, con)
        st_next_rk4 = st + (dt_mpc/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        g.append(st_next - st_next_rk4)

    # Терминальная стоимость
    s_end = X[6, N]
    ref_state_N = ca.vertcat(X_spline(s_end), Y_spline(s_end), yaw_spline(s_end), 15.0, 0, 0, s_end)
    state_error_N = X[:, N] - ref_state_N
    cost += ca.mtimes([state_error_N.T, Q_terminal, state_error_N]) 
    cost -= progress_weight * s_end # Бонус за пройденное расстояние

    # Создание солвера
    OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g_all = ca.vertcat(*g, *g_dev)
    
    nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g_all, 'p': P}
    opts = {
        'ipopt.print_level': 0, 
        'ipopt.max_iter': 100,      # Ограничим итерации для скорости
        'ipopt.tol': 1e-4, 
        'print_time': 0,
        'ipopt.warm_start_init_point': 'yes'
    }
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # 3. Границы переменных (Bounds)
    n_states = 7
    n_controls = 3
    n_vars = n_states * (N + 1) + n_controls * N
    
    lbx = -ca.inf * np.ones(n_vars)
    ubx = ca.inf * np.ones(n_vars)
    
    # Ограничения управления
    u_start_idx = n_states * (N + 1)
    for i in range(N):
        idx = u_start_idx + i * n_controls
        # Throttle [0, 1], Steering [-MAX, MAX], Brakes [0, 1]
        lbx[idx:idx+3] = [0, -MAX_STEER, 0]
        ubx[idx:idx+3] = [1, MAX_STEER, 1]
        
    # Ограничения s >= 0
    for k in range(N+1):
        s_idx = k * n_states + 6
        lbx[s_idx] = 0

    # Границы ограничений (Constraints Bounds)
    # Первые n_states * (N+1) уравнений динамики должны быть = 0
    # Следующие N ограничений CTE должны быть в пределах +/- max_deviation
    lbg = np.concatenate((np.zeros(n_states * (N + 1)), -max_deviation * np.ones(N)))
    ubg = np.concatenate((np.zeros(n_states * (N + 1)),  max_deviation * np.ones(N)))

    # 4. Симуляция
    if car_model is None:
        exit() # Выход, если нет C++ модуля

    car = car_model.Dynamic4WheelsModel()
    car.set_initial_state(yaw=np.pi / 2)
    
    # Инициализация переменных для симуляции
    x0_sol = np.zeros(n_vars) # Начальное приближение для солвера (все нули)
    
    vehicle_traj = {'x': [], 'y': [], 'yaw': []}
    u_apply = np.array([0.0, 0.0, 0.0]) 
    
    # Аккумулятор угла для непрерывности (Unwrap logic)
    current_yaw_accumulated = np.pi / 2
    prev_raw_yaw = np.pi / 2

    print("Старт симуляции...")
    
    for step in range(12000): # 30 секунд симуляции
        # Получение состояния из симулятора
        current_state = car.get_state()
        
        # --- ЛОГИКА UNWRAP ДЛЯ YAW ---
        raw_yaw = current_state.yaw
        dyaw = raw_yaw - prev_raw_yaw
        # Если скачок больше PI, корректируем аккумулятор
        if dyaw > np.pi:  dyaw -= 2*np.pi
        if dyaw < -np.pi: dyaw += 2*np.pi
        current_yaw_accumulated += dyaw
        prev_raw_yaw = raw_yaw
        # -----------------------------

        # Логгирование для графиков
        vehicle_traj['x'].append(current_state.X)
        vehicle_traj['y'].append(current_state.Y)
        vehicle_traj['yaw'].append(raw_yaw)

        # Запуск MPC только каждые mpc_steps шагов
        if step % mpc_steps == 0:
            # Найти s используя ОЧИЩЕННЫЙ путь
            s_curr = find_closest_s(current_state.X, current_state.Y, clean_path, s_knots_ref)
            
            # Формируем вектор состояния с НЕПРЕРЫВНЫМ углом
            x_curr_vec = np.array([
                current_state.X, current_state.Y, current_yaw_accumulated, 
                current_state.vx, current_state.vy, current_state.w, s_curr
            ])
            
            # Обновляем начальные условия для текущего шага MPC
            lbx[0:n_states] = x_curr_vec
            ubx[0:n_states] = x_curr_vec
            
            # Решаем
            try:
                sol = solver(x0=x0_sol, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_curr_vec)
                
                # Извлекаем управление
                sol_x = sol['x'].full().flatten()
                u_opt_all = sol_x[u_start_idx:]
                u_apply = u_opt_all[0:3]
                
                # Warm Start Strategy: Сохраняем решение как старт для следующего шага
                x0_sol = sol_x

            except Exception as e:
                print(f"Solver Error at step {step}: {e}")
                # Аварийное торможение
                u_apply = [0, 0, 1]

            if step % 100 == 0:
                print(f"Step {step}: s={s_curr:.1f}, v={current_state.vx:.1f}, u={u_apply}")

        # Применяем управление к машине (каждый тик dt_sim)
        # Важно: C++ модель скорее всего ожидает руль в радианах или normalized?
        # Если в радианах - подаем u_apply[1].
        u_cpp = car_model.ControlInfluence(float(u_apply[0]), float(u_apply[1]), float(u_apply[2]))
        car.update(u_cpp)

    # Визуализация
    plt.figure(figsize=(10, 8))
    plt.plot(clean_path[:,0], clean_path[:,1], 'g--', label='Reference Spline', linewidth=1)
    plt.plot(vehicle_traj['x'], vehicle_traj['y'], 'b-', label='MPC Trajectory', linewidth=2)
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.title("MPC Simulation Result")
    plt.savefig('simulation_mpc_fixed.png')
    plt.show()