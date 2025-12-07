import casadi as ca
import numpy as np

from sim.py_model.C16_CONTINENTAL_Tire_Data import *
from sim.py_model.constants import *


# [X, Y, yaw, vx, vy, w]
x = ca.SX.sym('x', 6)
# u = [throttle, steering_angle, brakes]
u = ca.SX.sym('u', 3)

X, Y, yaw, vx, vy, w = x[0], x[1], x[2], x[3], x[4], x[5]
throttle, steering_angle, brakes = u[0], u[1], u[2]

# Углы скольжения
ar = ca.if_else(vx==0, 0, ca.atan2(vy - lr*w, vx))
af = ca.if_else(vx==0, 0, ca.atan2(vy + lf*w, vx) - steering_angle)

# Продольные силы
Fdrv = throttle * Cm
Frrr = Crr * ca.tanh(vx)
Frrf = Crr * ca.tanh(vx)
Fdrag = Cd * vx**2
Fbf = brakes * Cbf * ca.tanh(vx)
Fbr = brakes * Cbr * ca.tanh(vx)

# Поперечные силы
Fry = 2 * Cx * ar
Ffy = 2 * Cx * af

# Полная динамика
Ftransversal = Fdrv - Frrr - Frrf*ca.cos(steering_angle) - Fdrag - Fbf*ca.cos(steering_angle) - Fbr - Ffy*ca.sin(steering_angle)
Flateral = -Frrf*ca.sin(steering_angle) - Fbf*ca.sin(steering_angle) + Fry + Ffy*ca.cos(steering_angle)
Lmoment = -Frrf*ca.sin(steering_angle)*lf - Fbf*ca.sin(steering_angle)*lf - Fry*lr + Ffy*ca.cos(steering_angle)*lf

# Производные состояния
dxdt = vx * ca.cos(yaw) - vy * ca.sin(yaw)
dydt = vx * ca.sin(yaw) + vy * ca.cos(yaw)
dyawdt = w
dvxdt = Ftransversal/m + vy*w
dvydt = Flateral/m - vx*w
dwdt  = Lmoment/Iz

xdot = ca.vertcat(dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt)

f = ca.Function('f', [x, u], [xdot])

if __name__ == "__main__":
    # Проверка работоспособности на тестовых данных
    # my_file = open('./controls/mpc/test_u.txt', 'r').read().split('\n')
    # n_steps = int(my_file[0])
    # us = [
    #     [float(j) for j in i.split(' ')] for i in my_file[1:]
    # ]
    # x_curr = np.zeros(6)

    # for i in range(n_steps):
    #     for _ in range(50):
    #         u_curr = np.array(us[i])
    #         # print(u_curr)
    #         xdot_val = f(x_curr, u_curr)
    #         x_curr = x_curr + 0.001 * xdot_val
    #     print(f"Step {i}: {x_curr}")
    
    N = 200
    U = ca.SX.sym('U', 3, N)
    X = ca.SX.sym('X', 6, N+1)
    Q = ca.diag([1, 10, 1, 1, 1, 1])
    R = ca.diag([1, 1, 10])
    cost = 0
    g = []
    x0 = np.zeros(6)
    x_ref = np.array([0, 10, 0, 0, 0, 0])
    nx, nu = 6, 3
    
    # Интеграция Эйлером (можно заменить на RK4 для точности)
    for k in range(N):
        x_next = X[:,k] + dt * f(X[:,k], U[:,k])
        g.append(X[:,k+1] - x_next)  # динамика
        cost += ca.mtimes([(X[:,k] - x_ref).T, Q, (X[:,k] - x_ref)]) \
                + ca.mtimes([U[:,k].T, R, U[:,k]])
    # print(cost)
    
    OPT_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
    g = ca.vertcat(*g)  # ограничения динамики

    nlp_prob = {'f': cost, 'x': OPT_variables, 'g': g}

    opts = {
        'ipopt.print_level': 0,
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    
    # Начальные значения
    u0 = np.zeros((3, N))
    x_curr = np.zeros(6)
    # x_curr = np.array([0, 0, 0, 100, 10, 0])
    # print(f'Current state{x_curr.shape}:\n{x_curr}')
    x0_guess = np.tile(x_curr.reshape(-1,1), (1, N+1))
    # print(f'First approximation of ref trajectory{x0_guess.shape}:\n{x0_guess}')
    x0_opt = np.concatenate([u0.flatten(), x0_guess.flatten()])  # nx * (1 + horizon) + nu * horizon
    # print(f'First approximation of opt variables{x0_opt.shape}:\n{x0_opt}')
    
    lbx = np.concatenate([np.tile([0.01, -0.5, 0], N), np.tile([-ca.inf]*nx, N+1)])  # lbx — lower bounds (нижние ограничения)
    # print(lbx)
    ubx = np.concatenate([np.tile([1, 0.5, 1], N), np.tile([ca.inf]*nx, N+1)])  # ubx — upper bounds (верхние ограничения)
    # print(ubx)
    
    lbg = np.zeros(nx * N); ubg = np.zeros(nx * N)

    sol = solver(x0=x0_opt, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    
    sol_u = sol['x'][:nu*N].full().reshape(nu, N)
    
    # print(sol)
    for t in range(100):
        sol = solver(x0=x0_opt, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        u_opt = sol['x'][:nu*N].full().reshape(nu, N)[:,0]
        # for to in range(50000):
        x_curr = x_curr + dt * f(x_curr, u_opt).full().flatten()
        # if 
        print([float(round(i, 3)) for i in x_curr], [float(round(i, 3)) for i in u_opt])
        # print(x_curr)
        # u0_guess = np.hstack([u_opt[:,1:], u_opt[:,-1:]]) 