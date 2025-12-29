from controls.mpc.test_start import *
from cpp_port import car_model
from sim.track_editor.create import *
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import numpy as np
from scipy.interpolate import CubicSpline

_, x_cl, y_cl = gen_double_P_track()
s_cl = compute_arc_length(x_cl, y_cl)
kappa_cl = compute_curvature(x_cl, y_cl, s_cl)
kappa_fun = build_kappa_function(s_cl, kappa_cl)
dt_sim = 0.001
predict_horizon = 30
mpc = FrenetMPC(kappa_fun, N=predict_horizon, dt=0.05)
x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
u_guess = np.array([1.0, 0.0, 0.0])
traj = [x]
controls = []
v_ref = 15.0  # Целевая скорость 15 м/с
car = car_model.Dynamic4WheelsModel()
car.set_initial_state(yaw=np.pi / 2)
time_vec = []
t = 0.0
# 1. Trajectory Re-mapping
dx_ds = np.gradient(x_cl, s_cl)
dy_ds = np.gradient(y_cl, s_cl)
theta_cl = np.arctan2(dy_ds, dx_ds)
sx = CubicSpline(s_cl, x_cl)
sy = CubicSpline(s_cl, y_cl)
stheta = CubicSpline(s_cl, theta_cl)
# C++ viz.
vehicle_traj = {'x': [], 'y': [], 'yaw': [], 'v': [], 'steering': [], 'cte': []}
cte_history = []
print("Запуск симуляции MPC...")

# Список для хранения предсказанных траекторий на каждом шаге
pred_trajs = []

for i in range(500):
    cartesian_state = car.get_state()
    # frenet_state = cartesian_to_frenet(
    # cartesian_state.x,
    # cartesian_state.y,
    # cartesian_state.taw,
    # cartesian_state.v,
    # mpc.
    # )
    # Решаем MPC
    X_pred, U_pred = mpc.solve(x, v_ref, u_guess)

    # Преобразуем предсказанные состояния в декартовы координаты для визуализации
    pred_s = X_pred[:, 0]
    pred_ey = X_pred[:, 1]
    pred_epsi = X_pred[:, 2]
    pred_s = np.clip(pred_s, s_cl[0], s_cl[-1])  # Избегаем выхода за пределы
    pred_x = sx(pred_s) - pred_ey * np.sin(stheta(pred_s))
    pred_y = sy(pred_s) + pred_ey * np.cos(stheta(pred_s))
    pred_trajs.append({'x': pred_x, 'y': pred_y})

    # Берем первое оптимальное управление
    u_control = U_pred[0]  # [throttle, steering, brakes]
    # Симуляция реального шага (Integration sub-stepping)
    # Используем ту же функцию физики для симуляции
    # Делаем несколько маленьких шагов, пока MPC думает "медленно"
    sim_steps = int(mpc.dt / dt_sim)  # например 0.05 / 0.01 = 5 шагов
    current_x_dm = ca.DM(x)
    u_dm = ca.DM(u_control)
    for _ in range(sim_steps):
        x_dot = mpc.f_dynamics(current_x_dm, u_dm)
        current_x_dm += x_dot * dt_sim
        u = car_model.ControlInfluence(u_dm[0], u_dm[1], u_dm[2])
        car.update(u)
        t += dt_sim  # Исправлено на dt_sim
    x = current_x_dm.full().flatten()
    c_state = frenet_to_cartesian(x[0], x[1], x[2], x[3], x[4], x[5], {'x': sx, 'y': sy, 'theta': stheta})
    # Сохраняем историю
    traj.append(x)
    controls.append(u_control)
    # Warm start для следующего шага
    u_guess = U_pred[1] if mpc.N > 1 else U_pred[0]
    if i % 20 == 0:
        print(f"Step {i}: s={x[0]:.2f}, v={x[3]:.2f}, ey={x[1]:.3f}, throt={u_control[0]:.2f}")

traj = np.array(traj)
controls = np.array(controls)
# print(x)
# ==========================================
# VISUALIZATION
# ==========================================
# 1. Trajectory Re-mapping
dx_ds = np.gradient(x_cl, s_cl)
dy_ds = np.gradient(y_cl, s_cl)
theta_cl = np.arctan2(dy_ds, dx_ds)
sx = CubicSpline(s_cl, x_cl)
sy = CubicSpline(s_cl, y_cl)
stheta = CubicSpline(s_cl, theta_cl)
traj_s = traj[:, 0]
traj_ey = traj[:, 1]
traj_epsi = traj[:, 2]
# Избегаем выхода за пределы сплайна (clamp s)
traj_s = np.clip(traj_s, s_cl[0], s_cl[-1])
x_traj = sx(traj_s) - traj_ey * np.sin(stheta(traj_s))
y_traj = sy(traj_s) + traj_ey * np.cos(stheta(traj_s))
# Вычисляем yaw для траектории
yaw_traj = stheta(traj_s) + traj_epsi

# Функция для рисования машины как четырехугольника
def draw_car(x, y, yaw, L=4.0, W=2.0):  # Размеры машины (длина 4м, ширина 2м)
    corners = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rotated = np.dot(corners, rot.T)
    car_x = rotated[:, 0] + x
    car_y = rotated[:, 1] + y
    return car_x, car_y

# Анимация
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x_cl, y_cl, 'k--', label='Центральная линия')
actual_line, = ax.plot([], [], 'r-', linewidth=2, label='Фактическая траектория')
pred_line, = ax.plot([], [], 'g--', label='Предсказанная траектория')
car_patch = Polygon([[0,0]], closed=True, color='b', alpha=0.5, label='Машина')
ax.add_patch(car_patch)
ax.set_xlabel('X [м]')
ax.set_ylabel('Y [м]')
ax.set_title('Визуализация движения машины с MPC')
ax.axis('equal')
ax.grid(True)
ax.legend()

def animate(i):
    # Фактическая траектория до текущего шага
    actual_line.set_data(x_traj[:i+1], y_traj[:i+1])
    # Предсказанная траектория на этом шаге
    pred_line.set_data(pred_trajs[i]['x'], pred_trajs[i]['y'])
    # Машина как четырехугольник
    car_x, car_y = draw_car(x_traj[i], y_traj[i], yaw_traj[i])
    car_patch.set_xy(list(zip(car_x, car_y)))
    return actual_line, pred_line, car_patch

anim = FuncAnimation(fig, animate, frames=len(traj)-1, interval=50, blit=True)

# Сохранение анимации (требует ffmpeg для mp4 или pillow для gif)
anim.save(f'mpc_animation_{predict_horizon}_{int(v_ref)}.gif', writer='pillow')

# Статичные графики (как раньше)
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(x_cl, y_cl, 'k--', label='Центральная линия')
plt.plot(x_traj, y_traj, 'r-', linewidth=2, label='MPC')
plt.ylabel('Y [м]')
plt.xlabel('X [м]')
plt.legend()
plt.title('Следование траектории')
plt.axis('equal')
plt.grid(True)
plt.subplot(4, 1, 2)
plt.plot(traj[:, 3], label='Скорости')
plt.axhline(v_ref, color='g', linestyle='--', label='Желаемая')
plt.ylabel('Скорость [м/с]')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 3)
plt.plot(controls[:, 0], label='Throttle')
plt.plot(controls[:, 1], label='Steering')
plt.plot(controls[:, 2], label='Brakes')
plt.ylabel('Управляющие воздействия')
plt.xlabel('Шаг симуляции')
plt.legend()
plt.grid(True)
plt.subplot(4, 1, 4)
plt.plot(traj[:, 1], label='Поперечное отклонение')
plt.ylabel('Поперечное отклонение [м]')
plt.xlabel('Шаг симуляции')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('mpc.png')