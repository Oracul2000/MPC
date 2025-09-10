from sim.py_model.model import *
from sim.track_editor.create import *

from controls.pid.pid import *
from controls.simple_steering_control.simple_steering_control import *
from controls.pure_pursuit.pure_pursuit import *
from controls.mpc.mpc import *

from matplotlib import pyplot as plt
import numpy as np


path, center_line_x, center_line_y = gen_double_P_track()
car = Dynamic4WheelsModel()
car.carState.body.yaw = np.pi / 2
mpc = MPCController(path, wheelbase=1.5, horizon=15, dt=0.05)

vehicle_traj = {'x': [], 'y': [], 'yaw': []}
dt = 0.05
v_ref = 10.0

for step in range(8000):
    x, y, yaw, v = car.getX(), car.getY(), car.getyaw(), car.getvx()
    throttle, steering = mpc.compute_control(x, y, yaw, v, v_ref)
    brakes = 0.0
    if throttle < 0:
        brakes = -throttle
        throttle = 0.0
    u = ControlInfluence(throttle, steering, brakes)
    car.updateRK4(u)
    
    vehicle_traj['x'].append(x)
    vehicle_traj['y'].append(y)
    vehicle_traj['yaw'].append(yaw)

plt.plot(center_line_x, center_line_y, c='green')
plt.plot(vehicle_traj['x'], vehicle_traj['y'])
plt.axis('equal')
plt.savefig('simulation_mpc.png')