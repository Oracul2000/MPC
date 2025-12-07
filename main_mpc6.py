# main_mpc_sim.py
import numpy as np
import casadi as ca
from matplotlib import pyplot as plt

# берем вашу динамику из main_mpc4 (файл должен быть в той же папке)
from main_mpc4 import get_dynamics_model

# берем генератор трека и C++ модель так же, как в main_pure_pursuit
from sim.track_editor.create import gen_double_P_track
from cpp_port import car_model

class MPCController:
    def __init__(self, center_x, center_y, N=20, dt_mpc=0.05, max_dev=0.2):
        self.N = N
        self.dt_mpc = dt_mpc
        self.max_deviation = max_dev

        # ========== подготовка сплайнов по arc-length ==========
        # вычислим кумулятивную длину (arc length) для параметизации s
        pts = np.stack((center_x, center_y), axis=1)
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        s_knots = np.concatenate(([0.0], np.cumsum(seg_lengths)))
        # если точки совпадают (маловероятно) — сделаем уникальными
        # создадим интерполянты (линейные — надёжнее)
        X_coeffs = center_x
        Y_coeffs = center_y
        # CasADi interpolant: аргументами должны быть numpy arrays
        self.X_spline = ca.interpolant('X_ref', 'linear', [s_knots], X_coeffs)
        self.Y_spline = ca.interpolant('Y_ref', 'linear', [s_knots], Y_coeffs)
        # yaw: можно считать как тангенсу направления между соседними точками
        # для CasADi создадим массив yaw_coeffs
        yaw_coeffs = np.zeros_like(X_coeffs)
        dx = np.gradient(X_coeffs)
        dy = np.gradient(Y_coeffs)
        yaw_coeffs = np.arctan2(dy, dx)
        self.yaw_spline = ca.interpolant('yaw_ref', 'linear', [s_knots], yaw_coeffs)

        # ========== создаём динамическую модель (CasADi) ==========
        self.f = get_dynamics_model()  # функция f(x,u) -> xdot, ожидает 7 states, 3 controls

        # размеры
        self.n_states = 7
        self.n_controls = 3

        # ========== формируем NLP аналогично вашему main_mpc4 ==========
        N = self.N
        X = ca.SX.sym('X', self.n_states, N + 1)
        U = ca.SX.sym('U', self.n_controls, N)
        P = ca.SX.sym('P', self.n_states)  # текущее состояние

        # веса (можно подстроить)
        Q = ca.diag([20, 20, 10, 5, 5, 5, 0])
        Q_terminal = 10 * Q
        R = ca.diag([10, 10, 1])
        progress_weight = 5.0

        cost = 0
        g_eq = []     # динамические равенства
        g_ineq = []   # ограничения отклонения (cte)

        for k in range(N):
            st = X[:, k]
            con = U[:, k]
            s_k = st[6]

            # reference по сплайну (CasADi выражения)
            X_ref = self.X_spline(s_k)
            Y_ref = self.Y_spline(s_k)
            yaw_ref = self.yaw_spline(s_k)
            ref_state = ca.vertcat(X_ref, Y_ref, yaw_ref, 0, 0, 0, s_k)

            state_error = st - ref_state
            cost += ca.mtimes([state_error.T, Q, state_error])
            cost += ca.mtimes([con.T, R, con])

            # cross-track error (cte)
            dx = st[0] - X_ref
            dy = st[1] - Y_ref
            theta_ref = yaw_ref
            cte = ca.cos(theta_ref) * dy - ca.sin(theta_ref) * dx
            # g_ineq should enforce -max_dev <= cte <= max_dev  => as two inequalities <=0
            g_ineq.append(cte - self.max_deviation)
            g_ineq.append(-cte - self.max_deviation)

            # RK4 integration
            st_next = X[:, k + 1]
            k1 = self.f(st, con)
            k2 = self.f(st + (dt_mpc/2) * k1, con)
            k3 = self.f(st + (dt_mpc/2) * k2, con)
            k4 = self.f(st + dt_mpc * k3, con)
            st_next_rk4 = st + (dt_mpc/6) * (k1 + 2*k2 + 2*k3 + k4)
            g_eq.append(st_next - st_next_rk4)

        # терминальный штраф + продвижение
        stN = X[:, N]
        X_ref_N = self.X_spline(stN[6])
        Y_ref_N = self.Y_spline(stN[6])
        yaw_ref_N = self.yaw_spline(stN[6])
        refN = ca.vertcat(X_ref_N, Y_ref_N, yaw_ref_N, 0, 0, 0, stN[6])
        state_error_N = stN - refN
        cost += ca.mtimes([state_error_N.T, Q_terminal, state_error_N])
        cost -= progress_weight * stN[6]

        # собираем NLP
        OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        g = ca.vertcat(*g_eq, *g_ineq)

        nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g, 'p': P}
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 400,
            'ipopt.tol': 1e-6
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        # ========== границы для переменных и ограничений ==========
        n_vars = self.n_states * (N + 1) + self.n_controls * N

        self.lbx = -1e20 * np.ones(n_vars)
        self.ubx = 1e20 * np.ones(n_vars)

        # ограничим управления: throttle [0,1], steer [-0.5,0.5], brakes [0,1]
        u_start_idx = self.n_states * (N + 1)
        for i in range(N):
            idx = u_start_idx + i * self.n_controls
            self.lbx[idx:idx + 3] = [0.0, -0.5, 0.0]
            self.ubx[idx:idx + 3] = [1.0, 0.5, 1.0]

        # ограничения на равенства динамики = 0
        n_eq = self.n_states * N
        n_ineq = 2 * N
        # lbg, ubg: первые n_eq entries must be 0; next 2*N entries: <=0 (we set ubg=0, lbg=-inf)
        large_neg = -1e20
        self.lbg = np.concatenate((np.zeros(n_eq), large_neg * np.ones(n_ineq)))
        self.ubg = np.concatenate((np.zeros(n_eq), np.zeros(n_ineq)))

        # начальная догадка
        self.x0_sol = np.zeros(n_vars)

    def compute_control(self, x_curr):
        """
        x_curr: numpy array shape (7,) = [X,Y,yaw,vx,vy,w,s]
        Возвращает u_apply = [throttle, steer, brakes]
        """
        try:
            sol = self.solver(x0=self.x0_sol, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=x_curr)
        except Exception as e:
            print("MPC solver failed:", e)
            # вернуть безопасное действие
            return np.array([0.0, 0.0, 0.0])

        sol_x = sol['x'].full().flatten()
        # сохранить для warm start
        self.x0_sol = sol_x
        # достаём первое управляющее действие
        u_start_idx = self.n_states * (self.N + 1)
        u_opt_all = sol_x[u_start_idx:]
        u_apply = u_opt_all[0:3]
        return u_apply

if __name__ == "__main__":
    # ======= подготовка трека и модели =======
    path, center_line_x, center_line_y = gen_double_P_track()
    # center_line_x/center_line_y — массивы координат центра пути
    # создаём автомобиль
    car = car_model.Dynamic4WheelsModel()
    car.set_initial_state(yaw=np.pi/2)

    # симуляция
    sim_dt = 0.001
    mpc_dt = 0.05
    mpc_hold_steps = int(mpc_dt / sim_dt)  # сколько сим. шагов держим оптимальный u
    N_mpc = 25

    mpc = MPCController(center_line_x, center_line_y, N=N_mpc, dt_mpc=mpc_dt, max_dev=0.3)

    # история
    hist_x, hist_y, hist_yaw = [], [], []
    hist_vx, hist_s = [], []

    # инициализация состояния s в модели — если модель не задаёт s, мы храним его отдельно
    s_val = 0.0
    # допустим, начальное состояние: X=0,Y=0,yaw=pi/2,vx=0
    # Убедитесь, что car.get_state() даёт vx в м/с и координаты согласованы с треком
    cur_u = np.array([0.0, 0.0, 0.0])
    for step in range(12000):
        st = car.get_state()
        x, y, yaw, vx = st.X, st.Y, st.body.yaw if hasattr(st, 'body') else st.yaw, st.vx
        # некоторые модели могут хранить yaw иначе; подстраховка:
        try:
            yaw = st.yaw
        except:
            pass

        # подготовим вектор состояния для MPC: [X,Y,yaw,vx,vy,w,s]
        # если car.get_state не отдаёт vy/w — можно поставить 0 или получить реальные значения
        vy = getattr(st, 'vy', 0.0)
        w = getattr(st, 'w', 0.0)
        # используем s_val как интеграл vx по времени (приближённо ds/dt = vx)
        if step == 0:
            s_val = 0.0
        else:
            s_val += vx * sim_dt

        x_curr = np.array([x, y, yaw, vx, vy, w, s_val])

        # пересчитываем MPC каждую mpc_hold_steps итераций
        if (step % mpc_hold_steps) == 0:
            u_mpc = mpc.compute_control(x_curr)
            # если throttle отрицательный — переводим в тормоз (наш MPC ограничивает throttle >=0)
            cur_u = u_mpc

        # формируем объект управления и обновляем модель
        u_obj = car_model.ControlInfluence(float(cur_u[0]), float(cur_u[1]), float(cur_u[2]))
        car.update(u_obj)

        # логирование
        hist_x.append(x); hist_y.append(y); hist_yaw.append(yaw)
        hist_vx.append(vx); hist_s.append(s_val)

        if step % 200 == 0:
            print(f"step {step}: X={x:.2f}, Y={y:.2f}, vx={vx:.2f}, s={s_val:.2f}, throttle={cur_u[0]:.3f}, steer={cur_u[1]:.3f}")

    # визуализация
    plt.figure(figsize=(8,6))
    plt.plot(center_line_x, center_line_y, 'g--', label='Center line')
    plt.plot(hist_x, hist_y, 'b-', label='Vehicle')
    plt.axis('equal')
    plt.legend()
    plt.savefig('mpc_sim_result.png')
    print("Saved mpc_sim_result.png")
