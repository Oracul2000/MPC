import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Попытка импорта, чтобы код не падал, если модуля нет при проверке
try:
    from cpp_port import car_model 
except ImportError:
    print("Внимание: Модуль cpp_port не найден. Симуляция не запустится.")
    # Создаем фиктивный класс для car_model, чтобы код не падал, если модуль отсутствует
    class FakeCarState:
        def __init__(self):
            self.X = 0.0
            self.Y = 0.0
            self.yaw = np.pi / 2
            self.vx = 0.0
            self.vy = 0.0
            self.w = 0.0
            
    class FakeControlInfluence:
        def __init__(self, throttle, steering, brakes):
            self.throttle = throttle
            self.steering = steering
            self.brakes = brakes

    class FakeDynamic4WheelsModel:
        def __init__(self):
            self.state = FakeCarState()
            self.time = 0.0
            self.dt_sim = 0.001

        def set_initial_state(self, X=0, Y=0, yaw=0, vx=0, vy=0, w=0, s=0):
            self.state.X = X
            self.state.Y = Y
            self.state.yaw = yaw
            self.state.vx = vx
            self.state.vy = vy
            self.state.w = w
            
        def get_state(self):
            return self.state

        def update(self, u_cpp):
            # Очень простая фиктивная динамика, чтобы симуляция длилась
            self.state.X += self.state.vx * np.cos(self.state.yaw) * self.dt_sim
            self.state.Y += self.state.vx * np.sin(self.state.yaw) * self.dt_sim
            self.state.yaw += self.state.w * self.dt_sim
            self.state.vx = min(15.0, self.state.vx + (u_cpp.throttle - u_cpp.brakes) * 10 * self.dt_sim)
            # Применим небольшой эффект руля к w
            self.state.w += u_cpp.steering * 0.5 * self.dt_sim * max(0.1, self.state.vx)
            self.time += self.dt_sim

    class FakeModule:
        Dynamic4WheelsModel = FakeDynamic4WheelsModel
        ControlInfluence = FakeControlInfluence

    car_model = FakeModule()

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
N = 10               # Горизонт прогноза
max_deviation = 2.0  # Максимальное отклонение от траектории
progress_weight = 10.0 # Вес прогресса 
MAX_STEER = 0.6      # Максимальный угол поворота колес (радианы)
V_REF_TARGET = 15.0  # Целевая скорость для горизонта предсказания
V_REF_TERMINAL = 15.0 # Целевая скорость на терминальном шаге

def get_dynamics_model():
    """ Создает CasADi функцию динамики автомобиля. """
    x = ca.SX.sym('x', 7)  # [X, Y, yaw, vx, vy, w, s]
    u = ca.SX.sym('u', 3)  # [throttle, steering, brakes]

    X, Y, yaw, vx, vy, w, s = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
    throttle, steering, brakes = u[0], u[1], u[2]

    # Углы скольжения
    ar = ca.if_else(vx < 0.1, 0, ca.atan2(vy - lr * w, vx + 1e-5))
    af = ca.if_else(vx < 0.1, 0, ca.atan2(vy + lf * w, vx + 1e-5) - steering)

    # Силы
    Fdrv = throttle * Cm
    Frrr = Crr * ca.tanh(vx)
    Frrf = Crr * ca.tanh(vx) 
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
    dsdt = vx 

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
    return points, x_merged, y_merged # Возвращаем raw_path, center_line_x, center_line_y

def get_trajectory_spline(path):
    """ Строит сплайн траектории. """
    diffs = np.diff(path, axis=0)
    lengths_temp = np.sqrt(np.sum(diffs**2, axis=1))
    mask = lengths_temp > 1e-2 
    cleaned_path = np.concatenate((path[0:1], path[1:][mask]))
    
    diffs = np.diff(cleaned_path, axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    s_knots = np.cumsum(np.concatenate(([0], lengths)))
    
    dx = np.gradient(cleaned_path[:, 0], s_knots)
    dy = np.gradient(cleaned_path[:, 1], s_knots)
    yaw_raw = np.arctan2(dy, dx)
    yaw_unwrapped = np.unwrap(yaw_raw) 

    X_spline = ca.interpolant('X_ref', 'bspline', [s_knots], cleaned_path[:, 0])
    Y_spline = ca.interpolant('Y_ref', 'bspline', [s_knots], cleaned_path[:, 1])
    yaw_spline = ca.interpolant('yaw_ref', 'bspline', [s_knots], yaw_unwrapped)
    
    return X_spline, Y_spline, yaw_spline, s_knots[-1], cleaned_path, s_knots

def find_closest_s_and_xy(x, y, cleaned_path, s_knots):
    """ Находит s и координаты ближайшей точки (x_ref, y_ref) на траектории. """
    dists = (cleaned_path[:,0] - x)**2 + (cleaned_path[:,1] - y)**2
    idx = np.argmin(dists)
    
    x_ref = cleaned_path[idx, 0]
    y_ref = cleaned_path[idx, 1]
    
    return s_knots[idx], x_ref, y_ref

if __name__ == "__main__":
    # 1. Подготовка модели и трассы
    f = get_dynamics_model()
    raw_path, center_line_x, center_line_y = gen_double_P_track()
    
    X_spline, Y_spline, yaw_spline, max_s_track, clean_path, s_knots_ref = get_trajectory_spline(raw_path)

    # 2. Формулировка MPC задачи (CasADi)
    U = ca.SX.sym('U', 3, N)      
    X = ca.SX.sym('X', 7, N + 1)  
    P = ca.SX.sym('P', 7)         

    Q = ca.diag([10, 10, 50, 0.1, 0, 0, 0]) 
    Q_terminal = 10 * Q
    R = ca.diag([1, 10, 50])            

    cost = 0
    g = []     
    g_dev = [] 

    st = X[:, 0]
    g.append(st - P) 

    for k in range(N):
        st = X[:, k]
        con = U[:, k]
        
        s_k = st[6]
        X_ref = X_spline(s_k)
        Y_ref = Y_spline(s_k)
        yaw_ref = yaw_spline(s_k)
        vx_ref = V_REF_TARGET # Целевая скорость
        
        ref_state = ca.vertcat(X_ref, Y_ref, yaw_ref, vx_ref, 0, 0, s_k)
        state_error = st - ref_state
        
        cost += ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([con.T, R, con])
        
        dx_err = st[0] - X_ref
        dy_err = st[1] - Y_ref
        cte = -ca.sin(yaw_ref) * dx_err + ca.cos(yaw_ref) * dy_err
        
        g_dev.append(cte) 

        # Динамика (RK4)
        st_next = X[:, k + 1]
        k1 = f(st, con); k2 = f(st + dt_mpc/2 * k1, con)
        k3 = f(st + dt_mpc/2 * k2, con); k4 = f(st + dt_mpc * k3, con)
        st_next_rk4 = st + (dt_mpc/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        g.append(st_next - st_next_rk4)

    # Терминальная стоимость
    s_end = X[6, N]
    ref_state_N = ca.vertcat(X_spline(s_end), Y_spline(s_end), yaw_spline(s_end), V_REF_TERMINAL, 0, 0, s_end)
    state_error_N = X[:, N] - ref_state_N
    cost += ca.mtimes([state_error_N.T, Q_terminal, state_error_N]) 
    cost -= progress_weight * s_end 

    # Солвер
    OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g_all = ca.vertcat(*g, *g_dev)
    nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g_all, 'p': P}
    opts = {'ipopt.print_level': 0, 'ipopt.max_iter': 100, 'ipopt.tol': 1e-4, 'print_time': 0, 'ipopt.warm_start_init_point': 'yes'}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    # Границы переменных
    n_states = 7; n_controls = 3
    n_vars = n_states * (N + 1) + n_controls * N
    lbx = -ca.inf * np.ones(n_vars); ubx = ca.inf * np.ones(n_vars)
    u_start_idx = n_states * (N + 1)
    for i in range(N):
        idx = u_start_idx + i * n_controls
        lbx[idx:idx+3] = [0, -MAX_STEER, 0]
        ubx[idx:idx+3] = [1, MAX_STEER, 1]
        s_idx = i * n_states + 6
        lbx[s_idx] = 0
    s_idx_N = N * n_states + 6
    lbx[s_idx_N] = 0

    lbg = np.concatenate((np.zeros(n_states * (N + 1)), -max_deviation * np.ones(N)))
    ubg = np.concatenate((np.zeros(n_states * (N + 1)),  max_deviation * np.ones(N)))
    print(lbg)
    print(ubg)
    exit()

    # 3. Симуляция
    car = car_model.Dynamic4WheelsModel()
    car.set_initial_state(yaw=np.pi / 2)
    
    x0_sol = np.zeros(n_vars) 
    u_apply = np.array([0.0, 0.0, 0.0]) 
    
    # Хранилища для графиков
    time_vec = []
    vehicle_traj = {'x': [], 'y': [], 'yaw': [], 'v': [], 'steering': [], 'throttle': [], 'brakes': [], 'cte': []}
    
    current_yaw_accumulated = np.pi / 2
    prev_raw_yaw = np.pi / 2
    
    TOTAL_SIM_STEPS = 12000
    LOG_INTERVAL = 100 # Логируем каждые 100 шагов
    
    print("Старт симуляции MPC...")
    
    for step in range(TOTAL_SIM_STEPS): 
        current_time = step * dt_sim
        current_state = car.get_state()
        
        # --- ЛОГИКА UNWRAP ДЛЯ YAW ---
        raw_yaw = current_state.yaw
        dyaw = raw_yaw - prev_raw_yaw
        if dyaw > np.pi:  dyaw -= 2*np.pi
        if dyaw < -np.pi: dyaw += 2*np.pi
        current_yaw_accumulated += dyaw
        prev_raw_yaw = raw_yaw
        # -----------------------------

        # Запуск MPC только каждые mpc_steps шагов
        if step % mpc_steps == 0:
            s_curr, _, _ = find_closest_s_and_xy(current_state.X, current_state.Y, clean_path, s_knots_ref)
            
            x_curr_vec = np.array([
                current_state.X, current_state.Y, current_yaw_accumulated, 
                current_state.vx, current_state.vy, current_state.w, s_curr
            ])
            
            # Обновление границ для начального состояния
            lbx[0:n_states] = x_curr_vec
            ubx[0:n_states] = x_curr_vec
            
            try:
                sol = solver(x0=x0_sol, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=x_curr_vec)
                sol_x = sol['x'].full().flatten()
                u_opt_all = sol_x[u_start_idx:]
                u_apply = u_opt_all[0:3]
                x0_sol = sol_x
            except Exception as e:
                # print(f"Solver Error at step {step}: {e}")
                u_apply = [0, 0, 1]

        # --- Вычисление CTE (Евклидово расстояние) ---
        s_curr, x_ref, y_ref = find_closest_s_and_xy(current_state.X, current_state.Y, clean_path, s_knots_ref)
        cte_euc = np.sqrt((current_state.X - x_ref)**2 + (current_state.Y - y_ref)**2)
        
        # Логгирование
        if step % LOG_INTERVAL == 0:
            vehicle_traj['x'].append(current_state.X)
            vehicle_traj['y'].append(current_state.Y)
            vehicle_traj['yaw'].append(raw_yaw)
            vehicle_traj['v'].append(current_state.vx)
            vehicle_traj['steering'].append(u_apply[1])
            vehicle_traj['throttle'].append(u_apply[0])
            vehicle_traj['brakes'].append(u_apply[2])
            vehicle_traj['cte'].append(cte_euc)
            time_vec.append(current_time)
            
            if step % 1000 == 0:
                print(f"Шаг {step}, t={current_time:.2f}с, v={current_state.vx:.2f} м/с, CTE={cte_euc:.3f} м, s={s_curr:.1f}")


        # Применяем управление
        u_cpp = car_model.ControlInfluence(float(u_apply[0]), float(u_apply[1]), float(u_apply[2]))
        car.update(u_cpp)

    # ===================================================================
    # ======================== Визуализация (На русском) ================
    # ===================================================================

    plt.rcParams['font.size'] = 10 # Уменьшим шрифт, чтобы все поместилось
    fig = plt.figure(figsize=(16, 10))

    # 1. Траектория
    plt.subplot(2, 3, 1)
    plt.plot(center_line_x, center_line_y, c='green', linewidth=2, linestyle='--', label='Центральная линия трассы')
    plt.plot(vehicle_traj['x'], vehicle_traj['y'], c='blue', label='Траектория машины (MPC)')
    plt.scatter(vehicle_traj['x'][0], vehicle_traj['y'][0], color='black', marker='o', s=50, label='Старт')
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.title('Траектория движения (MPC)')
    plt.xlabel('Координата X, м')
    plt.ylabel('Координата Y, м')
    plt.grid(True)
    

    # 2. Скорость
    plt.subplot(2, 3, 2)
    plt.plot(time_vec, vehicle_traj['v'], c='red', label='Фактическая скорость')
    plt.axhline(V_REF_TARGET, color='black', linestyle='--', label=f'v_ref (горизонт) = {V_REF_TARGET} м/с')
    plt.axhline(V_REF_TERMINAL, color='gray', linestyle=':', label=f'v_ref (терминал) = {V_REF_TERMINAL} м/с')
    plt.xlabel('Время, с')
    plt.ylabel('Скорость, м/с')
    plt.title('Скорость автомобиля (vₓ)')
    plt.legend()
    plt.grid(True)

    # 3. Угол руля
    plt.subplot(2, 3, 3)
    plt.plot(time_vec, np.degrees(vehicle_traj['steering']), c='purple', label='Угол руля')
    plt.axhline(np.degrees(MAX_STEER), color='grey', linestyle='--', linewidth=1, label=f'Max Steer = {np.degrees(MAX_STEER):.0f}°')
    plt.axhline(-np.degrees(MAX_STEER), color='grey', linestyle='--', linewidth=1)
    plt.xlabel('Время, с')
    plt.ylabel('Угол руля, °')
    plt.title('Управление рулём (Steering)')
    plt.legend()
    plt.grid(True)

    # 4. Cross-Track Error
    plt.subplot(2, 3, 4)
    plt.plot(time_vec, vehicle_traj['cte'], c='orange', label='Евклидово CTE')
    plt.axhline(max_deviation, color='red', linestyle='--', linewidth=1, label=f'Ограничение CTE = {max_deviation} м')
    plt.xlabel('Время, с')
    plt.ylabel('CTE, м')
    plt.title('Cross-Track Error (Отклонение от трассы)')
    plt.legend()
    plt.grid(True)

    # 5. Ускорение/Торможение
    plt.subplot(2, 3, 5)
    plt.plot(time_vec, vehicle_traj['throttle'], c='green', label='Акселератор [0, 1]')
    plt.plot(time_vec, vehicle_traj['brakes'], c='red', label='Тормоз [0, 1]')
    plt.xlabel('Время, с')
    plt.ylabel('Управляющее воздействие')
    plt.title('Продольное управление (Throttle/Brakes)')
    plt.legend()
    plt.grid(True)

    # 6. Угол рыскания
    plt.subplot(2, 3, 6)
    plt.plot(time_vec, np.degrees(vehicle_traj['yaw']), c='brown', label='Угол рыскания')
    plt.xlabel('Время, с')
    plt.ylabel('Угол рыскания, °')
    plt.title('Угол рыскания (Yaw)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('simulation_mpc_full_analysis_ru.png', dpi=300)
    plt.show()

    print("Симуляция завершена. Все графики сохранены в 'simulation_mpc_full_analysis_ru.png'")