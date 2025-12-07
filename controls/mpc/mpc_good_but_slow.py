import numpy as np
from scipy.optimize import minimize

class MPCController:
    def __init__(self, path, wheelbase=1.5, horizon=10, dt=0.05, cte_max=0.5):
        self.path = path
        self.wheelbase = wheelbase
        self.horizon = horizon
        self.dt = dt
        self.cte_max = cte_max  # Максимальный допустимый CTE
        self.Q = np.diag([10.0, 1000.0, 1.0, 10.0])  # Weights: x, y (high for CTE), yaw, v
        self.R = np.diag([0.1, 10.1])  # Weights for throttle, steering
        
        # Предвычисление длин дуг для пути (arc lengths)
        diffs = np.diff(self.path, axis=0)
        self.arc_lengths = np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))
        self.arc_lengths = np.insert(self.arc_lengths, 0, 0)  # Добавить 0 для первой точки

    def find_closest_point(self, x, y):
        distances = np.sqrt((self.path[:, 0] - x)**2 + (self.path[:, 1] - y)**2)
        idx = np.argmin(distances)
        return self.path[idx], idx

    def calculate_cte(self, x, y):
        # Приближённый CTE: расстояние до ближайшей точки на пути
        # (Для точности: проекция на сегмент между idx и idx+1)
        _, idx = self.find_closest_point(x, y)
        ref_x, ref_y = self.path[idx]
        cte = np.sqrt((ref_x - x)**2 + (ref_y - y)**2)
        # Знак CTE (опционально: лево/право от пути)
        # Для простоты берём абсолютное значение в constraint
        return cte

    def predict_states(self, x0, u):
        x, y, yaw, v = x0
        states = [(x, y, yaw, v)]
        for throttle, steering in u:
            x += v * np.cos(yaw) * self.dt
            y += v * np.sin(yaw) * self.dt
            yaw += (v / self.wheelbase) * np.tan(steering) * self.dt
            v += throttle * self.dt
            states.append((x, y, yaw, v))
        return np.array(states)

    def cost_function(self, u, x0, ref_states):
        u = u.reshape((self.horizon, 2))
        states = self.predict_states(x0, u)
        cost = 0.0
        for t in range(self.horizon):
            state_diff = states[t+1] - ref_states[t]  # Теперь 4 компонента: x, y, yaw, v
            cost += state_diff.T @ self.Q @ state_diff
            cost += u[t].T @ self.R @ u[t]
        return cost

    def cte_constraint(self, u, x0, ref_states):
        u = u.reshape((self.horizon, 2))
        states = self.predict_states(x0, u)
        constraints = []
        for t in range(self.horizon):
            cte = self.calculate_cte(states[t+1, 0], states[t+1, 1])
            constraints.append(self.cte_max - cte)  # CTE <= cte_max
        return np.array(constraints)

    def compute_control(self, x, y, yaw, v, max_v):
        x0 = np.array([x, y, yaw, v])
        _, idx = self.find_closest_point(x, y)
        current_arc = self.arc_lengths[idx]

        # Reference trajectory: продвижение по arc length на основе max_v
        ref_states = np.zeros((self.horizon, 4))  # x, y, yaw, v
        for t in range(self.horizon):
            desired_arc = current_arc + max_v * self.dt * (t + 1)
            ref_idx = np.searchsorted(self.arc_lengths, desired_arc)
            if ref_idx >= len(self.path):
                ref_idx = len(self.path) - 1
            # Интерполяция позиции
            if ref_idx > 0:
                prev_arc = self.arc_lengths[ref_idx - 1]
                frac = (desired_arc - prev_arc) / (self.arc_lengths[ref_idx] - prev_arc)
                ref_pos = (1 - frac) * self.path[ref_idx - 1] + frac * self.path[ref_idx]
            else:
                ref_pos = self.path[ref_idx]
            # Yaw: угол к следующей точке
            next_idx = min(ref_idx + 1, len(self.path) - 1)
            dy = self.path[next_idx, 1] - ref_pos[1]
            dx = self.path[next_idx, 0] - ref_pos[0]
            ref_yaw = np.arctan2(dy, dx)
            ref_states[t] = [ref_pos[0], ref_pos[1], ref_yaw, max_v]

        # Оптимизация
        u0 = np.zeros(self.horizon * 2)
        bounds = [(-1.0, 1.0), (-np.pi/4, np.pi/4)] * self.horizon  # Throttle, steering limits
        cons = {'type': 'ineq', 'fun': self.cte_constraint, 'args': (x0, ref_states)}
        result = minimize(
            self.cost_function, u0, args=(x0, ref_states),
            method='SLSQP', bounds=bounds, constraints=cons
        )
        u_opt = result.x.reshape((self.horizon, 2))
        return u_opt[0, 0], u_opt[0, 1]  # Return first control: throttle, steering