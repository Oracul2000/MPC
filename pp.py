from cpp_port import car_model  # <-- твой новый C++ модуль
from sim.track_editor.create import *
from controls.pid.pid import *
from controls.pure_pursuit.pure_pursuit import *
from matplotlib import pyplot as plt
import numpy as np

import matplotlib as mpl

# Глобальная настройка
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman'] + mpl.rcParams['font.serif']


path, center_line_x, center_line_y = gen_double_P_track()

pp = PurePursuitController(path, wheelbase=1.544, lookahead_base=1.0, 
                          lookahead_gain=0.2, max_lookahead=5.0)

pid = LongitudinalPID(kp=1.2, ki=0.5, kd=0.05, 
                     throttle_limits=(0.0, 1.0), integrator_limits=(-5, 5))

car = car_model.Dynamic4WheelsModel()
car.set_initial_state(yaw=np.pi / 2)
time_vec = []


vehicle_traj = {'x': [], 'y': [], 'yaw': [], 'v': [], 'steering': [], 'cte': []}
cte_history = []
dt = 0.001
v_ref = 7.0  # ~54 км/ч
t = 0.0



for step in range(12000):
    current_state = car.get_state()
    x, y = current_state.X, current_state.Y
    yaw = current_state.yaw
    v = current_state.vx
    steering = pp.compute_steering(x, y, yaw, v)
    throttle = pid.compute(v_ref, v, dt=dt)
    brakes = 0.0
    if throttle < 0:
        brakes = -throttle
        throttle = 0.0
    path_np = np.array(path)  # shape: (N, 2)
    pos = np.array([x, y])
    distances = np.linalg.norm(path_np - pos, axis=1)
    nearest_idx = np.argmin(distances)
    cte = distances[nearest_idx]  # евклидово расстояние до ближайшей точки
    if step % 100 == 0:
        vehicle_traj['x'].append(x)
        vehicle_traj['y'].append(y)
        vehicle_traj['yaw'].append(yaw)
        vehicle_traj['v'].append(v)
        vehicle_traj['steering'].append(steering)
        vehicle_traj['cte'].append(cte)
        time_vec.append(t)
    u = car_model.ControlInfluence(throttle, steering, brakes)
    car.update(u)
    t += dt
    if step % 1000 == 0:
        print(f"Step {step}, t={t:.2f}s, v={v:.2f} m/s, CTE={cte:.3f} m")


fig = plt.figure(figsize=(16, 10))
plt.subplot(2, 3, 1)
plt.plot(center_line_x, center_line_y, c='green', linewidth=2, label='Центральная линия')
plt.plot(vehicle_traj['x'], vehicle_traj['y'], c='blue', label='Траектория машины')
plt.axis('equal')
plt.legend()
plt.title('Траектория движения')
plt.grid(True)
plt.subplot(2, 3, 2)
plt.plot(time_vec, vehicle_traj['v'], c='red')
plt.axhline(v_ref, color='black', linestyle='--', label=f'v_ref = {v_ref} м/с')
plt.xlabel('Время, с')
plt.ylabel('Скорость, м/с')
plt.title('Скорость автомобиля')
plt.legend()
plt.grid(True)
plt.subplot(2, 3, 3)


plt.plot(time_vec, np.degrees(vehicle_traj['steering']), c='purple')


plt.xlabel('Время, с')


plt.ylabel('Угол руля, °')


plt.title('Управление рулём (Pure Pursuit)')


plt.grid(True)





# 4. Cross-Track Error


plt.subplot(2, 3, 4)


plt.plot(time_vec, vehicle_traj['cte'], c='orange')


plt.xlabel('Время, с')


plt.ylabel('CTE, м')


plt.title('Cross-Track Error (отклонение от трассы)')


plt.grid(True)





# 5–6. Можно добавить ещё что-нибудь, например yaw или ускорения


plt.subplot(2, 3, 5)


plt.plot(time_vec, np.degrees(vehicle_traj['yaw']), c='brown')


plt.xlabel('Время, с')


plt.ylabel('Угол рыскания, °')


plt.title('Угол рыскания (yaw)')


plt.grid(True)





plt.tight_layout()


plt.savefig('simulation_pure_pursuit_full_analysis.png', dpi=300)


plt.show()





print("Симуляция завершена. Все графики сохранены в 'simulation_pure_pursuit_full_analysis.png'")