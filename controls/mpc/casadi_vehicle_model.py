import casadi as ca
import numpy as np

from constants import *


def get_dynamics_model():
    """ Создает CasADi функцию динамики автомобиля. """
    x = ca.SX.sym('x', 6)  # [X, Y, yaw, vx, vy, w]
    u = ca.SX.sym('u', 3)  # [throttle, steering, brakes]

    X, Y, yaw, vx, vy, w = x[0], x[1], x[2], x[3], x[4], x[5]
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

    xdot = ca.vertcat(dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt)
    return ca.Function('f', [x, u], [xdot])


# def get_frenet_model():
#     x = ca.SX.sym('x', 6)  # s, ey, epsi, v, beta, r
#     u = ca.SX.sym('u', 3)  # throttle, steering, brakes
    
#     # Unpack X & U
#     s = x[0]
#     ey = x[1]
#     epsi = x[2]
#     v = x[3]
#     beta = x[4]
#     r = x[5]

#     throttle = u[0]
#     steering = u[1]
#     brakes = u[2]
    
    

if __name__ == '__main__':
    print(
        '''
         //              \\
        ((  TEST OF MODEL )) 
         \\               //
        '''
    )
    
    f = get_dynamics_model()
    state = np.array([0 for _ in range(6)])
    
    test_control = open('controls/mpc/test_u.txt', 'rt').read().split('\n')[1:]
    for i in test_control:
        throttle, steering, brakes = list(map(float, i.split(' ')))
        u = np.array([throttle, steering, brakes])
        for j in range(50):
            x_dot_t = f(state, u)
            state += x_dot_t * dt
        
        print(state)
        
        