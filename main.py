from sim.py_model.model import *
from sim.track_editor.create import *

from controls.pid.pid import *
from controls.simple_steering_control.simple_steering_control import *

from matplotlib import pyplot as plt
import numpy as np

model = Dynamic4WheelsModel()
model.carState.body.yaw = np.pi / 2
center_line, center_line_x, center_line_y = gen_P_track()

speed_pid = PID(kp=1.0, ki=1.0, kd=0.00, target=3)

iterations_by_one_step = 50

vehicle_traj = {
    'x': [],
    'y': []
}

for step in range(200):
    current_speed = model.carState.body.vx
    # print(f'vx = {current_speed}', end='\t')
    throttle_brake = speed_pid.update(current_speed, dt)
    # print(f'throttle_brake = {throttle_brake}', end='\t')
    throttle = max(0, throttle_brake)
    brakes = max(0, -throttle_brake)
    # print(f'({throttle}, {brakes})')
    steering = steering_controller(model, center_line, 2)
    input = ControlInfluence(throttle, steering, brakes)
    for i in range(iterations_by_one_step):
        model.updateRK4(input)
    print(model)
    vehicle_traj['x'].append(model.getX())
    vehicle_traj['y'].append(model.getY())
    
plt.plot(center_line_x, center_line_y, c='green')
plt.plot(vehicle_traj['x'], vehicle_traj['y'])
plt.axis('equal')
plt.savefig('simulation.png')
