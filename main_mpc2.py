import casadi as ca
import numpy as np

# ---------------------------
# Параметры (взяты из твоего C++/замены)
# ---------------------------
m = 1500.0
Iz = 2500.0
lf = 1.2
lr = 1.4
Cm = 2000.0
Crr = 100.0
Cd = 0.3
Cbf = 2000.0
Cbr = 1500.0
Cx = 50000.0

# симуляционный шаг
dt = 0.05

# ---------------------------
# Построение CasADi функции динамики f(x,u) -> xdot
# ---------------------------
def build_dynamics():
    x = ca.SX.sym('x', 6)   # [X, Y, yaw, vx, vy, w]
    u = ca.SX.sym('u', 3)   # [throttle, steering, brakes]

    X, Y, yaw, vx, vy, w = x[0], x[1], x[2], x[3], x[4], x[5]
    throttle, steering, brakes = u[0], u[1], u[2]

    # Защита от нуля для vx в atan2
    eps = 1e-3
    vx_safe = ca.fmax(vx, eps)

    # slip angles (как в C++)
    ar = ca.atan2(vy - lr * w, vx_safe)
    af = ca.atan2(vy + lf * w, vx_safe) - steering

    # продольные силы
    Fdrv = throttle * Cm
    Frrr = Crr * ca.tanh(vx_safe)
    Frrf = Crr * ca.tanh(vx_safe)
    Fdrag = Cd * vx_safe**2
    Fbf = brakes * Cbf * ca.tanh(vx_safe)
    Fbr = brakes * Cbr * ca.tanh(vx_safe)

    # поперечные силы (упрощённая форма в C++)
    Fry = 2.0 * Cx * ar
    Ffy = 2.0 * Cx * af

    # силы/момент (как в твоём C++)
    Ftransversal = (Fdrv
                    - Frrr
                    - Frrf * ca.cos(steering)
                    - Fdrag
                    - Fbf * ca.cos(steering)
                    - Fbr
                    - Ffy * ca.sin(steering))

    Flateral = (- Frrf * ca.sin(steering)
                - Fbf * ca.sin(steering)
                + Fry
                + Ffy * ca.cos(steering))

    Lmoment = (- Frrf * ca.sin(steering) * lf
               - Fbf * ca.sin(steering) * lf
               - Fry * lr
               + Ffy * ca.cos(steering) * lf)

    # производные
    dxdt = vx * ca.cos(yaw) - vy * ca.sin(yaw)
    dydt = vx * ca.sin(yaw) + vy * ca.cos(yaw)
    dyawdt = w
    dvxdt = (Ftransversal / m) + vy * w
    dvydt = (Flateral / m) - vx * w
    dwdt = Lmoment / Iz

    xdot = ca.vertcat(dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt)

    return ca.Function('f', [x, u], [xdot])

f = build_dynamics()

# ---------------------------
# MPC настройки
# ---------------------------
N = 30              # горизонт (кол-во шагов)
nx = 6
nu = 3
dt_mpc = dt         # интегр. шаг в MPC

# веса
Q_diag = np.array([200.0, 200.0, 1.0, 10.0, 10.0, 10.0])  # штраф по состоянию
R_diag = np.array([1.0, 5.0, 1.0])                    # штраф по управлению
Qf_diag = Q_diag * 10.0                                # терминальный вес

Q = ca.diag(ca.SX(Q_diag))
R = ca.diag(ca.SX(R_diag))
Qf = ca.diag(ca.SX(Qf_diag))

# целевое состояние (X=5,Y=5, другие желаемые нули)
x_target = np.array([5.0, 5.0, 0.0, 0.0, 0.0, 0.0])

# ---------------------------
# Построение NLP (multiple shooting)
# ---------------------------
# decision variables
X = ca.SX.sym('X', nx, N+1)
U = ca.SX.sym('U', nu, N)

P = ca.SX.sym('P', nx + nx)  # параметры: текущее состояние + цель (мы передаём целью тоже x_target)

cost = 0
g = []

for k in range(N):
    st = X[:, k]
    con = U[:, k]
    # error to target (используем параметр P для гибкости)
    target = P[nx:]  # target state passed in P
    e = st - target
    cost = cost + ca.mtimes([e.T, Q, e]) + ca.mtimes([con.T, R, con])

    # динамика (explicit Euler)
    k1 = f(st, con)
    st_next = X[:, k+1]
    st_next_euler = st + dt_mpc * k1
    g.append(st_next - st_next_euler)

# терминальная стоимость
eT = X[:, N] - P[nx:]
cost = cost + ca.mtimes([eT.T, Qf, eT])

# сбор NLP
OPT_variables = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
g = ca.vertcat(*g)

nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g, 'p': P}

# опции для IPOPT (правильная структура)
opts = {
    'ipopt': {
        'print_level': 0,
        'max_iter': 200,
        'tol': 1e-6,
        'linear_solver': 'mumps'
    },
    'print_time': False
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

# ---------------------------
# Ограничения и начальные приближения
# ---------------------------
total_vars = nx*(N+1) + nu*N

# bounds на переменные
lbx = -1e20 * np.ones((total_vars,))
ubx =  1e20 * np.ones((total_vars,))

# bounds на U (throttle [0,1], steering [-0.5,0.5], brakes [0,1])
u_start = nx*(N+1)
for i in range(N):
    idx = int(u_start + i*nu)
    lbx[idx:idx+3] = [0.0, -1.0, 0.0]
    ubx[idx:idx+3] = [1.0,  1.0, 1.0]

# равенства динамики
lbg = np.zeros(nx*N)
ubg = np.zeros(nx*N)

# ---------------------------
# Симуляция receding horizon
# ---------------------------
# начальное состояние
x_curr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# цель (передадим в P)
x_ref = x_target.copy()

# начальное приближение для x0_sol
x0_sol = np.zeros((total_vars,))
# инициализация X_guess как повтор x_curr
for k in range(N+1):
    x0_sol[k*nx:(k+1)*nx] = x_curr
# небольшое ненулевое управление вначале
for k in range(N):
    x0_sol[u_start + k*nu : u_start + (k+1)*nu] = [0.2, 0.0, 0.0]

print("Start receding horizon simulation...")

# симулируем T_steps шагов MPC
T_steps = 80
for tstep in range(T_steps):
    # заблокировать первые nx переменных (X[:,0] == x_curr)
    lbx[0:nx] = x_curr
    ubx[0:nx] = x_curr

    # параметр: текущее состояние + целевое
    p_val = np.concatenate((x_curr, x_ref))

    try:
        sol = solver(x0=x0_sol, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=p_val)
    except Exception as e:
        print("Solver failed at step", tstep, "error:", e)
        break

    sol_x = sol['x'].full().flatten()

    # извлекаем первое управление
    u_opt = sol_x[u_start:u_start+nu]
    throttle_cmd, steer_cmd, brakes_cmd = u_opt.tolist()

    # применяем в «реальном мире» явным Эйлером на dt
    xdot = f(x_curr, u_opt).full().flatten()
    x_next = x_curr + dt * xdot
    x_curr = x_next

    # warm start: shift X and U из решения
    X_sol = sol_x[0: nx*(N+1)]
    U_sol = sol_x[nx*(N+1):]

    # сдвигаем X_guess: берем X[1..N], добавляем последний как повтор
    X_guess = np.zeros_like(X_sol)
    X_guess[0:(N)*nx] = X_sol[nx:(N+1)*nx]
    X_guess[(N)*nx:(N+1)*nx] = X_sol[(N)*nx:(N+1)*nx]  # повтор последнего
    # U_guess: сдвигаем на один (берём U[1..], и добавляем последний)
    U_guess = np.zeros_like(U_sol)
    if N > 1:
        U_guess[0:(N-1)*nu] = U_sol[nu:(N)*nu]
        U_guess[(N-1)*nu: N*nu] = U_sol[(N-1)*nu: N*nu]
    else:
        U_guess[:] = U_sol

    x0_sol = np.concatenate([X_guess, U_guess])

    # лог
    print(f"t={tstep:02d} X={x_curr[0]:.3f} Y={x_curr[1]:.3f} yaw={x_curr[2]:.3f} vx={x_curr[3]:.3f} throttle={throttle_cmd:.3f} steer={steer_cmd:.3f} brakes={brakes_cmd:.3f}")

    # если близко к цели — заканчиваем
    pos_err = np.linalg.norm(x_curr[0:2] - x_ref[0:2])
    speed = np.linalg.norm(x_curr[3:5])
    if pos_err < 0.05 and speed < 0.1:
        print("Reached target sufficiently close at step", tstep)
        break

print("Finished.")
