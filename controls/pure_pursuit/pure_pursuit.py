import numpy as np


class PurePursuitController:
    """
    Pure Pursuit для кинематической модели.
    path: iterable of waypoints (Nx2 array or list of (x, y))
    lookahead: либо число >0 (м в пространстве), либо строк 'k*v' to use k*speed; здесь реализовано как параметр lookahead_base и lookahead_gain
    Перед вызовом compute_steering передайте текущее положение (x,y), yaw (rad), скорость v (для адаптивного lookahead).
    Возвращает steering angle (rad). Угол положителен — поворот влево (обычно).
    """
    def __init__(self, path, wheelbase=1.0, lookahead_base=2.0, lookahead_gain=0.1, max_lookahead=10.0):
        self.path = np.array(path)
        self.wheelbase = wheelbase
        self.lookahead_base = float(lookahead_base)
        self.lookahead_gain = float(lookahead_gain)
        self.max_lookahead = float(max_lookahead)

        # вспомогательное: индекс ближайшей точки на траектории, чтобы искать цель вперед
        self.last_closest_idx = 0
        
        self.viz_goal = None

    def _distance(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def _find_goal_index(self, pos, lookahead_dist):
        """
        Находит индекс первой точки траектории, которая на расстоянии >= lookahead_dist
        от текущей позиции при проходе вдоль пути, начиная от last_closest_idx.
        """
        N = len(self.path)
        if N == 0:
            return None
        # начнем с ближайшей точки (быстро)
        dists = np.hypot(self.path[:,0]-pos[0], self.path[:,1]-pos[1])
        # начнем с ближайшей точки глобально, но чтобы не откатываться назад, возьмём max между ней и last idx
        nearest_idx = int(np.argmin(dists))
        start = min(nearest_idx, self.last_closest_idx)
        # пробежим по точкам начиная со start, найдём первую, выполняющую условие
        for i in range(start, N):
            if np.hypot(self.path[i,0]-pos[0], self.path[i,1]-pos[1]) >= lookahead_dist:
                self.last_closest_idx = i
                return i
        # если не нашли — вернём последнюю точку
        self.last_closest_idx = N-1
        return N-1

    def compute_steering(self, x, y, yaw, v=0.0, lookahead_override=None, pure_pursuit_limit=np.deg2rad(30.0)):
        """
        Возвращает steering angle (rad). yaw в радианах.
        lookahead_override: если указано, используется вместо формулы lookahead_base + lookahead_gain * v
        pure_pursuit_limit: ограничение выходного руления (рад)
        """
        pos = (x, y)
        # адаптивный lookahead
        if lookahead_override is None:
            Ld = self.lookahead_base + self.lookahead_gain * abs(v)
        else:
            Ld = float(lookahead_override)
        Ld = min(Ld, self.max_lookahead)
        # найти индекс цели
        goal_idx = self._find_goal_index(pos, Ld)
        goal = self.path[goal_idx]
        
        self.viz_goal = goal

        # Преобразуем цель в локальные координаты передней оси (assume rear axle or reference point at (x,y))
        # Если yaw — угол направления движения относительно оси x
        dx = goal[0] - x
        dy = goal[1] - y
        # координаты цели в локальной системе (x вперед, y left)
        local_x =  np.cos(-yaw) * dx - np.sin(-yaw) * dy
        local_y =  np.sin(-yaw) * dx + np.cos(-yaw) * dy

        if local_x == 0 and local_y == 0:
            return 0.0

        # Pure Pursuit geometric steering: curvature kappa = 2*y / Ld^2  (approx, if using rear axle reference)
        # steering angle delta = atan(L * kappa)
        kappa = 2.0 * local_y / (Ld**2)
        delta = np.arctan(self.wheelbase * kappa)

        # ограничение руления
        if pure_pursuit_limit is not None:
            delta = np.clip(delta, -pure_pursuit_limit, pure_pursuit_limit)

        return float(delta)
