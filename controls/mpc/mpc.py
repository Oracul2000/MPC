import numpy as np
from scipy.optimize import minimize

class MPCController:
    def __init__(self, path, wheelbase=1.5, horizon=10, dt=0.05):
        self.path = path
        self.wheelbase = wheelbase
        self.horizon = horizon
        self.dt = dt
        self.Q = np.diag([10.0, 10.0, 1.0])  # Weights for x, y, yaw
        self.R = np.diag([0.1, 10.1])         # Weights for throttle, steering

    def find_closest_point(self, x, y):
        distances = np.sqrt((self.path[:, 0] - x)**2 + (self.path[:, 1] - y)**2)
        idx = np.argmin(distances)
        return self.path[idx], idx

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
            state_diff = states[t+1, :3] - ref_states[t, :3]
            cost += state_diff.T @ self.Q @ state_diff
            cost += u[t].T @ self.R @ u[t]
        return cost

    def compute_control(self, x, y, yaw, v, v_ref):
        x0 = np.array([x, y, yaw, v])
        _, idx = self.find_closest_point(x, y)
        
        # Reference trajectory (simple: follow path points with target velocity)
        ref_states = np.zeros((self.horizon, 3))
        for t in range(self.horizon):
            ref_idx = min(idx + t, len(self.path) - 1)
            ref_states[t] = [self.path[ref_idx, 0], self.path[ref_idx, 1], np.arctan2(
                self.path[min(ref_idx+1, len(self.path)-1), 1] - self.path[ref_idx, 1],
                self.path[min(ref_idx+1, len(self.path)-1), 0] - self.path[ref_idx, 0]
            )]

        # Optimize
        u0 = np.zeros(self.horizon * 2)
        bounds = [(-0.5, 0.5), (-np.pi/6, np.pi/6)] * self.horizon  # Throttle, steering limits
        result = minimize(
            self.cost_function, u0, args=(x0, ref_states),
            method='SLSQP', bounds=bounds
        )
        u_opt = result.x.reshape((self.horizon, 2))
        return u_opt[0, 0], u_opt[0, 1], result  # Return first control: throttle, steering
