from cpp_port import car_model
from sim.track_editor.create import gen_double_P_track
from controls.mpc.mpc import MPCController

from matplotlib import pyplot as plt
import numpy as np

path, center_line_x, center_line_y = gen_double_P_track()

car = car_model.Dynamic4WheelsModel()
car.set_initial_state(yaw=np.pi / 2)

mpc = MPCController(path, wheelbase=1.5, horizon=15, dt=0.05)

vehicle_traj = {'x': [], 'y': [], 'yaw': [], 'v': []}
dt = 0.05
v_ref = 30.0

steps_per_control_update = int(dt / 0.001)

print(f"Controller dt: {dt}s, Model dt: 0.001s. Running {steps_per_control_update} model steps per control step.")

for step in range(200):
    current_state = car.get_state()
    x, y, yaw, v = current_state.X, current_state.Y, current_state.yaw, current_state.vx
    
    vehicle_traj['x'].append(x)
    vehicle_traj['y'].append(y)
    vehicle_traj['yaw'].append(yaw)
    vehicle_traj['v'].append(v)

    throttle, steering = mpc.compute_control(x, y, yaw, v, v_ref)
    
    brakes = 0.0
    if throttle < 0:
        brakes = -throttle
        throttle = 0.0
        
    u = car_model.ControlInfluence(throttle, steering, brakes)
    
    for _ in range(steps_per_control_update):
        car.update(u)
        
    if step % 100 == 0:
        print(x, y, yaw, throttle, steering, v)

plt.figure(figsize=(12, 8))
plt.plot(center_line_x, center_line_y, 'g--', label='Track Centerline')
plt.plot(vehicle_traj['x'], vehicle_traj['y'], 'b-', label='MPC Trajectory')
plt.plot(vehicle_traj['v'])
plt.title('MPC speed')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.savefig('simulation_mpc_cpp_15_2.png')
plt.show()
