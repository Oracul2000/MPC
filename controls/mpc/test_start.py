import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from casadi_frenet_model import *


dt_sim = 0.001  # Шаг интегрирования симуляции


def generate_centerline():
    t = np.linspace(0, 20, 500)
    x = t
    y = 1.0 * np.sin(0.8 * t)
    return x, y

def compute_arc_length(x, y):
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.zeros(len(x))
    s[1:] = np.cumsum(ds)
    return s

def compute_curvature(x, y, s):
    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    # Защита от деления на ноль
    den = (dx**2 + dy**2)**1.5
    kappa = (dx * ddy - dy * ddx) / (den + 1e-8) 
    return kappa

def build_kappa_function(s, kappa):
    return ca.interpolant('kappa', 'linear', [s], kappa)

class FrenetMPC:
    def __init__(self, kappa_fun, N=30, dt=0.5):
        self.N = N
        self.dt = dt
        self.kappa_fun = kappa_fun

        # States: [s, ey, epsi, v, beta, r]
        # Controls: [throttle, steering, brakes]
        nx, nu = 6, 3 

        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)

        # --- ЗАГРУЗКА ВНЕШНЕЙ МОДЕЛИ ---
        # Получаем функцию f(x, u) -> xdot из твоего файла
        # Эта функция уже содержит всю физику (Pacejka, Drag, etc.)
        self.f_dynamics = get_frenet_dynamics_model(kappa_fun)

        # Multiple shooting setup
        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)
        P = ca.SX.sym('P', nx + 1) # [x0, v_ref]
        
        x0 = P[:nx]
        v_ref = P[nx]

        # --- ВЕСОВЫЕ КОЭФФИЦИЕНТЫ (Tuning) ---
        # Q: [s, ey, epsi, v, beta, r]
        # Высокий штраф на ey и epsi для точности, средний на v для поддержания скорости
        Q = ca.diag(ca.vertcat(0.0, 0.0, 0.0, 100.0, 20.0, 1.0)) 
        
        # R: [throttle, steering, brakes]
        # Штрафуем использование управления для плавности
        R = ca.diag(ca.vertcat(1.0, 1.0, 1.0))

        cost = 0
        g = [X[:, 0] - x0] # Начальное ограничение

        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]

            # Ошибка состояния (reference: ey=0, epsi=0, v=v_ref, beta=0, r=0)
            # s не штрафуем (0.0 в Q), так как мы хотим ехать вперед, а не стоять на месте s_ref
            # но можно добавить penalty за deviation от s_ref, если нужно строгое временное следование
            state_error = xk - ca.vertcat(xk[0], 0.0, 0.0, v_ref, 0.0, 0.0)
            
            cost += state_error.T @ Q @ state_error
            cost += uk.T @ R @ uk

            # Дискретизация (Euler integration)
            # self.f_dynamics возвращает xdot
            x_dot = self.f_dynamics(xk, uk)
            for _ in range(50):
                x_next = xk + self.dt * x_dot
            
            g.append(X[:, k+1] - x_next)

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # --- ГРАНИЦЫ (BOUNDS) ---
        inf = ca.inf
        
        # Границы для U: [throttle, steering, brakes] для N шагов
        # throttle: [0, 1]
        # steering: [-0.5, 0.5] рад (~28 градусов)
        # brakes:   [0, 1]
        lb_u = np.tile([0.0, -2.0, 0.0], N)
        ub_u = np.tile([1.0,  2.0, 3.0], N)

        # Границы для X (nx * (N+1))
        # v >= 0
        lb_x = -inf * np.ones(nx * (N + 1))
        ub_x = inf * np.ones(nx * (N + 1))
        
        # Устанавливаем v >= 0 для всех шагов прогноза
        for i in range(N + 1):
            lb_x[i * nx + 3] = 0.0
            
        # Применяем ограничения для каждого шага прогноза
        for i in range(N + 1):
            # Индекс начала состояния для шага i
            idx = i * nx
            
            # 1. Ограничение на ey (индекс 1)
            # Мы ставим жесткие рамки: машина не может выехать за пределы +/- 0.5 метра
            lb_x[idx + 1] = -0.5
            ub_x[idx + 1] = 0.5
            
            # 2. Ограничение на скорость v (индекс 3) >= 0
            lb_x[idx + 3] = 0.0

        self.lbx = ca.vertcat(lb_x, lb_u)
        self.ubx = ca.vertcat(ub_x, ub_u)

        nlp = {'x': opt_vars, 'f': cost, 'g': ca.vertcat(*g), 'p': P}
        
        # IPOPT Options
        opts = {
            'ipopt.print_level': 0,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'print_time': 0,
            'ipopt.warm_start_init_point': 'yes'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.nx, self.nu = nx, nu

    def solve(self, x0, v_ref, u_guess):
        # x0 должна быть вектором (6,)
        if isinstance(x0, ca.DM):
            x0 = x0.full().flatten()
            
        P = np.concatenate([x0, [v_ref]])

        # Warm start
        x_init = np.tile(x0, (self.N+1, 1)).flatten()
        
        # u_guess теперь должен быть (N, 3) или совместимый
        # if u_guess.ndim == 1 and u_guess.shape[0] == self.nu:
        u_init = np.tile(u_guess, (self.N, 1)).flatten()
        # else:
        #      # Fallback
        #      u_init = np.zeros(self.N * self.nu)

        sol = self.solver(
            x0=np.concatenate([x_init, u_init]),
            lbx=self.lbx, ubx=self.ubx,
            lbg=0, ubg=0, p=P
        )

        opt = sol['x'].full().flatten()
        
        # Извлекаем решение
        # X: (N+1, nx)
        # U: (N, nu)
        idx_u_start = self.nx * (self.N + 1)
        X_opt = opt[:idx_u_start].reshape(self.N+1, self.nx)
        U_opt = opt[idx_u_start:].reshape(self.N, self.nu)

        return X_opt, U_opt

# ==========================================
# MAIN LOOP
# ==========================================

# 1. Build Path
x_cl, y_cl = generate_centerline()
s_cl = compute_arc_length(x_cl, y_cl)
kappa_cl = compute_curvature(x_cl, y_cl, s_cl)
kappa_fun = build_kappa_function(s_cl, kappa_cl)

# 2. Init MPC
# dt в MPC (0.05) больше, чем dt симуляции (0.01), чтобы смотреть дальше
mpc = FrenetMPC(kappa_fun, N=30, dt=0.01) 

# Initial state: [s, ey, epsi, v, beta, r]
x = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]) 

# Initial guess inputs [throttle, steering, brakes]
u_guess = np.array([1.0, 0.0, 0.0]) 

traj = [x]
controls = []

v_ref = 15.0 # Целевая скорость 15 м/с

print("Запуск симуляции MPC...")
for i in range(200):
    # Решаем MPC
    X_pred, U_pred = mpc.solve(x, v_ref, u_guess)
    
    # Берем первое оптимальное управление
    u_control = U_pred[0] # [throttle, steering, brakes]
    
    # Симуляция реального шага (Integration sub-stepping)
    # Используем ту же функцию физики для симуляции
    # Делаем несколько маленьких шагов, пока MPC думает "медленно"
    sim_steps = int(mpc.dt / dt_sim) # например 0.05 / 0.01 = 5 шагов
    
    current_x_dm = ca.DM(x)
    u_dm = ca.DM(u_control)
    
    for _ in range(sim_steps):
        x_dot = mpc.f_dynamics(current_x_dm, u_dm)
        current_x_dm += x_dot * dt_sim
        
    x = current_x_dm.full().flatten()
    
    # Сохраняем историю
    traj.append(x)
    controls.append(u_control)
    
    # Warm start для следующего шага
    u_guess = U_pred[1] if mpc.N > 1 else U_pred[0]
    
    if i % 20 == 0:
        print(f"Step {i}: s={x[0]:.2f}, v={x[3]:.2f}, ey={x[1]:.3f}, throt={u_control[0]:.2f}")

traj = np.array(traj)
controls = np.array(controls)

# ==========================================
# VISUALIZATION
# ==========================================

# 1. Trajectory Re-mapping
dx_ds = np.gradient(x_cl, s_cl)
dy_ds = np.gradient(y_cl, s_cl)
theta_cl = np.arctan2(dy_ds, dx_ds)

sx = CubicSpline(s_cl, x_cl)
sy = CubicSpline(s_cl, y_cl)
stheta = CubicSpline(s_cl, theta_cl)

traj_s = traj[:, 0]
traj_ey = traj[:, 1]
traj_epsi = traj[:, 2]

# Избегаем выхода за пределы сплайна (clamp s)
traj_s = np.clip(traj_s, s_cl[0], s_cl[-1])

x_traj = sx(traj_s) - traj_ey * np.sin(stheta(traj_s))
y_traj = sy(traj_s) + traj_ey * np.cos(stheta(traj_s))

# 2. Plots
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(x_cl, y_cl, 'k--', label='Centerline')
plt.plot(x_traj, y_traj, 'r-', linewidth=2, label='MPC Trajectory')
plt.ylabel('Y [m]')
plt.xlabel('X [m]')
plt.legend()
plt.title('Path Tracking')
plt.axis('equal')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(traj[:, 3], label='Velocity')
plt.axhline(v_ref, color='g', linestyle='--', label='Reference')
plt.ylabel('Speed [m/s]')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(controls[:, 0], label='Throttle')
plt.plot(controls[:, 1], label='Steering')
plt.plot(controls[:, 2], label='Brakes')
plt.ylabel('Control Inputs')
plt.xlabel('Step')
plt.legend()
plt.grid(True)

plt.tight_layout()
# plt.show()

plt.savefig('./controls/mpc/test_start.png')