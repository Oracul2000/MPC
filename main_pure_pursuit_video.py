from sim.py_model.model import *
from sim.track_editor.create import *

from controls.pid.pid import *
from controls.simple_steering_control.simple_steering_control import *
from controls.pure_pursuit.pure_pursuit import *

import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter

from matplotlib import pyplot as plt
import numpy as np

# model = Dynamic4WheelsModel()
# model.carState.body.yaw = -np.pi / 2
path, center_line_x, center_line_y = gen_P_track()

# контроллеры
pp = PurePursuitController(path, wheelbase=1.5, lookahead_base=1.0, lookahead_gain=0.2, max_lookahead=2.0)
pid = LongitudinalPID(kp=1.2, ki=0.5, kd=0.05, throttle_limits=(0.0, 1.0), integrator_limits=(-5, 5))

vehicle_traj = {'x': [], 'y': [], 'yaw': [], 'lookahead_x': [], 'lookahead_y': []}
lf, lr, b = 0.721, 0.823, 0.669

# модель автомобиля
car = Dynamic4WheelsModel()
car.carState.body.yaw = np.pi / 2

# параметры симуляции
dt = 0.05
v_ref = 10.0  # целевая скорость, м/с (~54 км/ч)

for step in range(7500):
    x, y, yaw, v = car.getX(), car.getY(), car.getyaw(), car.getvx()
    
    # lateral control (руль)
    steering = pp.compute_steering(x, y, yaw, v)
    
    # longitudinal control (газ/тормоз)
    throttle = pid.compute(v_ref, v, dt=dt)
    brakes = 0.0
    if throttle < 0:  # если отрицательный выход — интерпретируем как торможение
        brakes = -throttle
        throttle = 0.0
    
    # применяем к динамической модели
    u = ControlInfluence(throttle, steering, brakes)
    car.updateRK4(u)
    
    if step % 20 == 0:
        print(car)


        vehicle_traj['x'].append(x)
        vehicle_traj['y'].append(y)
        vehicle_traj['yaw'].append(yaw)
        vehicle_traj['lookahead_x'].append(pp.viz_goal[0])
        vehicle_traj['lookahead_y'].append(pp.viz_goal[1])
        
# plt.plot(center_line_x, center_line_y, c='green')
# plt.plot(vehicle_traj['x'], vehicle_traj['y'])
# plt.axis('equal')
# plt.savefig('simulation.png')

fig, ax = plt.subplots()
ax.plot(center_line_x, center_line_y, c='green')
ax.set_aspect('equal')

base_points = np.array([[-lr, -b], [-lr, b], [lf, b], [lf, -b]])
car_patch = patches.Polygon(base_points, closed=True, color='blue')
ax.add_patch(car_patch)

# Placeholder for lookahead arrow, not added to plot yet
lookahead_arrow = None

def update(frame):
    global lookahead_arrow
    x, y, yaw = vehicle_traj['x'][frame], vehicle_traj['y'][frame], vehicle_traj['yaw'][frame]
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotated = base_points @ rot.T + [x, y]
    car_patch.set_xy(rotated)
    
    # Remove previous arrow if it exists
    if lookahead_arrow is not None:
        lookahead_arrow.remove()
    
    # Create new lookahead arrow
    lx, ly = vehicle_traj['lookahead_x'][frame], vehicle_traj['lookahead_y'][frame]
    dx, dy = lx - x, ly - y
    lookahead_arrow = ax.arrow(x, y, dx, dy, color='red', head_width=0.1, head_length=0.2)
    
    return car_patch, lookahead_arrow

anim = FuncAnimation(fig, update, frames=len(vehicle_traj['x']), blit=True)
writer = FFMpegWriter(fps=30)
anim.save('simulation_pure_pursuit.mp4', writer=writer)