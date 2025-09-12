from cpp_port import car_model  # <-- ИЗМЕНЕНИЕ: Используем новый C++ модуль
from sim.track_editor.create import *

from controls.pid.pid import *
from controls.simple_steering_control.simple_steering_control import *
from controls.pure_pursuit.pure_pursuit import *

from matplotlib import pyplot as plt
import numpy as np

# model = Dynamic4WheelsModel()
# model.carState.body.yaw = -np.pi / 2
path, center_line_x, center_line_y = gen_double_P_track()

# контроллеры
pp = PurePursuitController(path, wheelbase=1.5, lookahead_base=1.0, lookahead_gain=0.2, max_lookahead=5.0)
pid = LongitudinalPID(kp=1.2, ki=0.5, kd=0.05, throttle_limits=(0.0, 1.0), integrator_limits=(-5, 5))

# модель автомобиля
car = car_model.Dynamic4WheelsModel()
car.set_initial_state(yaw=np.pi / 2) 

vehicle_traj = {
    'x': [],
    'y': [],
    'yaw': []
}

# параметры симуляции
dt = 0.001
v_ref = 15.0  # целевая скорость, м/с (~54 км/ч)

for step in range(12000):
    current_state = car.get_state()
    x, y, yaw, v = current_state.X, current_state.Y, current_state.yaw, current_state.vx
    
    # lateral control (руль)
    steering = pp.compute_steering(x, y, yaw, v)
    
    # longitudinal control (газ/тормоз)
    throttle = pid.compute(v_ref, v, dt=dt)
    brakes = 0.0
    if throttle < 0:  # если отрицательный выход — интерпретируем как торможение
        brakes = -throttle
        throttle = 0.0
    
    # применяем к динамической модели
    u = car_model.ControlInfluence(throttle, steering, brakes)
    car.update(u)
    
    if step % 100 == 0:
        print(car)


    vehicle_traj['x'].append(x)
    vehicle_traj['y'].append(y)
    vehicle_traj['yaw'].append(yaw)
        
plt.plot(center_line_x, center_line_y, c='green')
plt.plot(vehicle_traj['x'], vehicle_traj['y'])
# plt.plot(vehicle_traj['v'])
plt.axis('equal')
plt.savefig('simulation_pure_pursuit_2.png')
