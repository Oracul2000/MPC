import casadi as ca
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt

# Предполагаем, что constants.py находится рядом
from constants import * 


def generate_centerline():
    t = np.linspace(0, 10, 200)
    x = t
    y = 1.5 * np.sin(0.8 * t)
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
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return kappa


def build_kappa_function(s, kappa):
    return ca.interpolant('kappa', 'linear', [s], kappa)


def get_frenet_dynamics_model(kappa_fun):
    x = ca.SX.sym('x', 6)
    s, ey, epsi, v, beta, r = x[0], x[1], x[2], x[3], x[4], x[5]

    # Controls: [throttle, steering, brakes]
    u = ca.SX.sym('u', 3)
    throttle, steering, brakes = u[0], u[1], u[2]

    k = kappa_fun(s)

    vx = v * ca.cos(beta)
    vy = v * ca.sin(beta)
    w = r
    ar = ca.if_else(vx < 0.1, 0, ca.atan2(vy - lr * w, vx + 1e-5))
    af = ca.if_else(vx < 0.1, 0, ca.atan2(vy + lf * w, vx + 1e-5) - steering)

    Fdrv = throttle * Cm
    Frrr = Crr * ca.tanh(vx)       # Сопротивление качения (зад)
    Frrf = Crr * ca.tanh(vx)       # Сопротивление качения (перед)
    Fdrag = Cd * vx**2             # Аэродинамическое сопротивление
    Fbf = brakes * Cbf * ca.tanh(vx) # Тормоза (перед)
    Fbr = brakes * Cbr * ca.tanh(vx) # Тормоза (зад)

    D_tire = 1.0 * m * 9.81 / 2
    Fry = D_tire * ca.tanh(2 * Cx * ar / D_tire)
    Ffy = D_tire * ca.tanh(2 * Cx * af / D_tire)

    Fx = Fdrv - Frrr - Frrf * ca.cos(steering) - Fdrag - Fbf * ca.cos(steering) - Fbr - Ffy * ca.sin(steering)
    Fy = -Frrf * ca.sin(steering) - Fbf * ca.sin(steering) + Fry + Ffy * ca.cos(steering)
    
    Lmoment = -Frrf * ca.sin(steering) * lf - Fbf * ca.sin(steering) * lf - Fry * lr + Ffy * ca.cos(steering) * lf

    dvdt = (Fx * ca.cos(beta) + Fy * ca.sin(beta)) / m
    dbetadt = ((-Fx * ca.sin(beta) + Fy * ca.cos(beta)) / (m * (v + 1e-5))) - r
    drdt = Lmoment / Iz
    dsdt = v * ca.cos(epsi + beta) / (1 - k * ey)    
    deydt = v * ca.sin(epsi + beta)
    depsidt = r - k * dsdt

    xdot = ca.vertcat(dsdt, deydt, depsidt, dvdt, dbetadt, drdt)
    
    return ca.Function('f_frenet', [x, u], [xdot], ['x', 'u'], ['xdot'])


if __name__ == '__main__':
    # 1. Генерируем траекторию и Kappa функцию (как в твоем коде FrenetMPC)
    # x_cl, y_cl = generate_centerline()
    
    test_traj = open('controls/mpc/test_y.txt', 'rt').read().split('\n')[1:]
    x_cl = []
    y_cl = []
    for i in test_traj:
        poses = i.split('{')[1].split(' ')
        x_cl.append(float(poses[0]))
        y_cl.append(float(poses[1]))
        
    x_cl = np.array(x_cl); y_cl = np.array(y_cl)
    
    s_cl = compute_arc_length(x_cl, y_cl)
    kappa_cl = compute_curvature(x_cl, y_cl, s_cl)
    
    # Создаем интерполянт для использования внутри CasADi графа
    kappa_fun = ca.interpolant('kappa', 'linear', [s_cl], kappa_cl)

    # 2. Получаем новую модель
    f = get_frenet_dynamics_model(kappa_fun)

    # 3. Тест
    # State: [s=0, ey=0.5, epsi=0, v=10, beta=0, r=0]
    # Начинаем с небольшим смещением ey=0.5 и скоростью 10 м/с
    state = ca.DM([0, 0, 0, 0, 0, 0])
    
    traj = []
    
    test_control = open('controls/mpc/test_u.txt', 'rt').read().split('\n')[1:]
    for i in test_control:
        throttle, steering, brakes = list(map(float, i.split(' ')))
        u = np.array([throttle, steering, brakes])
        for j in range(50):
            # x_dot_t = f(state, u)
            # state += x_dot_t * dt
            """
		state k1 = Derivatives(input, carState, af, ar, kappaf, kappar);
		state k2 = Derivatives(input, carState + k1 * (h / 2), af, ar, kappaf, kappar);
		state k3 = Derivatives(input, carState + k2 * (h / 2), af, ar, kappaf, kappar);
		state k4 = Derivatives(input, carState + k3 * h, af, ar, kappaf, kappar);
            """
            k1 = f(state, u)
            k2 = f(state + k1 * (dt / 2), u)
            k3 = f(state + k2 * (dt / 2), u)
            k4 = f(state + k3 * dt, u)
            state = state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)
            traj.append(state)
    traj = np.array(traj)
    
        
        # ----- Преобразование в Cartesian координаты -----
    dx_ds = np.gradient(x_cl, s_cl)
    dy_ds = np.gradient(y_cl, s_cl)
    theta_cl = np.arctan2(dy_ds, dx_ds)

    sx = CubicSpline(s_cl, x_cl)
    sy = CubicSpline(s_cl, y_cl)
    stheta = CubicSpline(s_cl, theta_cl)

    traj_s = traj[:, 0]
    traj_ey = traj[:, 1]
    traj_epsi = traj[:, 2]

    x_traj = sx(traj_s) - traj_ey * np.sin(stheta(traj_s))
    y_traj = sy(traj_s) + traj_ey * np.cos(stheta(traj_s))
    psi_traj = stheta(traj_s) + traj_epsi
    
    plt.plot(x_cl, y_cl, label='Исходные данные')
    plt.plot(x_traj, y_traj, label='Интегрирование во Frenet')
    plt.legend(['Исходные данные', 'Интегрирование во Frenet'])
    plt.savefig('./controls/mpc/test.png')