import casadi as ca
import numpy as np


def gen_double_P_track(x1=5, y1=25, r1=5):
    """Генерация точек трассы."""
    first_sector = {'x': np.zeros(y1 - r1), 'y': np.linspace(0, y1 - r1, y1 - r1)}
    sector2points_qty = 50 * r1
    theta2sector = np.pi - np.linspace(2 * np.pi / sector2points_qty, 3 * np.pi / 2 - 2 * np.pi / sector2points_qty, sector2points_qty - 1)
    second_sector = {'x': r1 * np.cos(theta2sector) + r1, 'y': r1 * np.sin(theta2sector) + y1 - r1}
    third_sector = {'x': np.ones(y1 - 2 * r1) * r1, 'y': np.linspace(0, y1 - 2 * r1, y1 - 2 * r1)}
    x_merged = np.concatenate((first_sector['x'], second_sector['x'], third_sector['x']))
    y_merged = np.concatenate((first_sector['y'], second_sector['y'], third_sector['y'][::-1]))
    x_merged = np.concatenate((x_merged, x_merged + r1))
    y_merged = np.concatenate((y_merged, -y_merged))
    points = np.stack((x_merged, y_merged), axis=1)
    return points, x_merged, y_merged




def get_trajectory_spline(path):
    """
    Строит сплайн траектории и возвращает ОЧИЩЕННЫЕ данные для синхронизации.
    """
    # 1. Очистка от дубликатов
    diffs = np.diff(path, axis=0)
    lengths_temp = np.sqrt(np.sum(diffs**2, axis=1))
    mask = lengths_temp > 1e-2 # Порог чуть выше машинной точности
    cleaned_path = np.concatenate((path[0:1], path[1:][mask]))
    # 2. Пересчет длин дуг (s) для очищенного пути
    diffs = np.diff(cleaned_path, axis=0)
    lengths = np.sqrt(np.sum(diffs**2, axis=1))
    s_knots = np.cumsum(np.concatenate(([0], lengths)))
    # 3. Расчет угла с unwrap (ВАЖНО для замкнутых трасс)
    dx = np.gradient(cleaned_path[:, 0], s_knots)
    dy = np.gradient(cleaned_path[:, 1], s_knots)
    yaw_raw = np.arctan2(dy, dx)
    yaw_unwrapped = np.unwrap(yaw_raw) # Убирает скачки pi -> -pi


    # 4. Создание сплайнов
    X_spline = ca.interpolant('X_ref', 'bspline', [s_knots], cleaned_path[:, 0])
    Y_spline = ca.interpolant('Y_ref', 'bspline', [s_knots], cleaned_path[:, 1])
    yaw_spline = ca.interpolant('yaw_ref', 'bspline', [s_knots], yaw_unwrapped)
    # Возвращаем сплайны + данные, на которых они построены
    return X_spline, Y_spline, yaw_spline, s_knots[-1], cleaned_path, s_knots




def find_closest_s(x, y, cleaned_path, s_knots):
    """
    Находит s, используя те же данные, что и сплайн.
    """
    dists = (cleaned_path[:,0] - x)**2 + (cleaned_path[:,1] - y)**2
    idx = np.argmin(dists)
    return idx, s_knots[idx]