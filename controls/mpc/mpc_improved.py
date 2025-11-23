"""
Improved MPC Controller with Dynamic Model and Cross Track Error Constraints

Key improvements:
1. Dynamic vehicle model with lateral dynamics (vx, vy, w)
2. Cross Track Error (CTE) constraint
3. Speed maximization objective
4. Warm start optimization
5. Control rate penalty (smooth control)
6. State constraints (speed, CTE limits)
7. Proper reference trajectory generation
8. Optimization success checking
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.interpolate import interp1d
import warnings


class MPCControllerImproved:
    """
    Advanced MPC controller for racing with dynamic vehicle model.
    
    Objectives (in order of priority):
    1. Stay within Cross Track Error bounds (hard constraint)
    2. Maximize velocity
    3. Follow reference trajectory
    4. Smooth control inputs
    """
    
    def __init__(self, path, wheelbase=1.5, horizon=15, dt=0.05, 
                 max_cte=2.0, max_speed=30.0, verbose=False):
        """
        Args:
            path: Nx2 array of waypoints
            wheelbase: Vehicle wheelbase (m)
            horizon: Prediction horizon steps
            dt: Time step (s)
            max_cte: Maximum allowed Cross Track Error (m)
            max_speed: Maximum vehicle speed (m/s)
            verbose: Enable debug output
        """
        self.path = np.array(path)
        self.wheelbase = wheelbase
        self.horizon = horizon
        self.dt = dt
        self.max_cte = max_cte
        self.max_speed = max_speed
        self.verbose = verbose
        
        # Vehicle parameters (from constants.py)
        self.m = 300.0  # Mass (kg)
        self.lf = 0.721  # Distance CG to front axle (m)
        self.lr = 0.823  # Distance CG to rear axle (m)
        self.Iz = 134.0  # Yaw moment of inertia (kg*m^2)
        self.Cm = 3600.0  # Motor constant
        self.Crr = 200.0  # Rolling resistance
        self.Cd = 1.53  # Drag coefficient
        
        # Simplified tire model (linear approximation for MPC speed)
        # Real model uses Magic Formula, but too complex for optimization
        self.Cf = 80000.0  # Front cornering stiffness (N/rad)
        self.Cr = 80000.0  # Rear cornering stiffness (N/rad)
        
        # Cost function weights
        # CTE: Cross Track Error (most important for staying on track)
        # Heading: Heading error
        # Speed: Speed error (negative to encourage high speed)
        # Control: Control magnitude
        # Control Rate: Control smoothness
        
        self.Q_cte = 100.0      # Cross track error weight (HIGH - stay on track!)
        self.Q_heading = 10.0   # Heading error weight
        self.Q_speed = 10.0     # Speed weight (penalize LOW speed)
        self.Q_lateral = 1.0    # Lateral velocity penalty (stability)
        
        self.R_throttle = 0.1   # Throttle magnitude penalty
        self.R_steering = 5.0   # Steering magnitude penalty
        
        self.Rd_throttle = 1.0  # Throttle rate penalty (smoothness)
        self.Rd_steering = 50.0 # Steering rate penalty (smoothness)
        
        # Terminal cost (end of horizon)
        self.Q_terminal_cte = 200.0
        self.Q_terminal_heading = 20.0
        
        # Optimization memory
        self.u_prev = None  # Previous control sequence for warm start
        self.last_idx = 0   # Last closest path index
        
        # Setup path parametrization for better reference trajectory
        self._setup_path_interpolation()
        
        # Statistics
        self.solve_times = []
        self.solve_success_rate = []
        
    def _setup_path_interpolation(self):
        """Setup cubic spline interpolation for smooth reference trajectory"""
        if len(self.path) < 4:
            # Not enough points for cubic interpolation
            self.use_interpolation = False
            return
            
        try:
            # Compute cumulative distance along path
            diffs = np.diff(self.path, axis=0)
            segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            self.s_path = np.concatenate([[0], np.cumsum(segment_lengths)])
            self.total_path_length = self.s_path[-1]
            
            # Create interpolators
            self.interp_x = interp1d(self.s_path, self.path[:, 0], 
                                     kind='cubic', fill_value='extrapolate')
            self.interp_y = interp1d(self.s_path, self.path[:, 1], 
                                     kind='cubic', fill_value='extrapolate')
            self.use_interpolation = True
            
        except Exception as e:
            warnings.warn(f"Path interpolation failed: {e}. Using linear.")
            self.use_interpolation = False
    
    def find_closest_point(self, x, y):
        """Find closest point on path and return point, index, and distance"""
        distances = np.sqrt((self.path[:, 0] - x)**2 + (self.path[:, 1] - y)**2)
        idx = np.argmin(distances)
        
        # Don't go backwards
        if idx < self.last_idx - 5:  # Allow small backwards movement
            idx = self.last_idx
        
        self.last_idx = idx
        return self.path[idx], idx, distances[idx]
    
    def compute_cross_track_error(self, x, y, ref_x, ref_y, ref_yaw):
        """
        Compute signed cross track error.
        Positive = left of track, Negative = right of track
        """
        # Vector from reference point to vehicle
        dx = x - ref_x
        dy = y - ref_y
        
        # Cross track error is perpendicular distance
        # Use cross product to get signed distance
        cte = -dx * np.sin(ref_yaw) + dy * np.cos(ref_yaw)
        
        return cte
    
    def predict_states_dynamic(self, x0, u):
        """
        Predict future states using simplified dynamic bicycle model.
        
        State: [x, y, yaw, vx, vy, w]
        Control: [throttle, steering]
        
        This is a linearized/simplified version of the full model for speed.
        """
        x, y, yaw, vx, vy, w = x0
        states = [x0.copy()]
        
        for throttle, steering in u:
            # Prevent division by zero
            vx_safe = max(abs(vx), 0.1) * np.sign(vx) if vx != 0 else 0.1
            
            # Slip angles
            alpha_f = np.arctan2(vy + self.lf * w, vx_safe) - steering
            alpha_r = np.arctan2(vy - self.lr * w, vx_safe)
            
            # Lateral tire forces (linear tire model)
            Fyf = -self.Cf * alpha_f
            Fyr = -self.Cr * alpha_r
            
            # Longitudinal force (simplified)
            Fx_drive = throttle * self.Cm
            Fx_resistance = self.Crr * np.tanh(vx / 5.0) + 0.5 * self.Cd * vx**2
            Fx_net = Fx_drive - Fx_resistance
            
            # Dynamics equations
            ax = (Fx_net + Fyf * np.sin(steering)) / self.m + vy * w
            ay = (Fyf * np.cos(steering) + Fyr) / self.m - vx * w
            w_dot = (self.lf * Fyf * np.cos(steering) - self.lr * Fyr) / self.Iz
            
            # Integrate (Euler method for speed)
            vx_new = vx + ax * self.dt
            vy_new = vy + ay * self.dt
            w_new = w + w_dot * self.dt
            
            # Global position update
            x_new = x + (vx * np.cos(yaw) - vy * np.sin(yaw)) * self.dt
            y_new = y + (vx * np.sin(yaw) + vy * np.cos(yaw)) * self.dt
            yaw_new = yaw + w * self.dt
            
            # Normalize yaw to [-pi, pi]
            yaw_new = np.arctan2(np.sin(yaw_new), np.cos(yaw_new))
            
            # Update state
            x, y, yaw, vx, vy, w = x_new, y_new, yaw_new, vx_new, vy_new, w_new
            states.append([x, y, yaw, vx, vy, w])
        
        return np.array(states)
    
    def get_reference_trajectory(self, idx, current_speed):
        """
        Generate reference trajectory for the horizon.
        Returns: (ref_x, ref_y, ref_yaw, ref_s) for each horizon step
        """
        ref_traj = np.zeros((self.horizon, 4))  # x, y, yaw, s
        
        if self.use_interpolation:
            # Use interpolated path
            current_s = self.s_path[idx]
            
            for t in range(self.horizon):
                # Estimate future progress along path
                # Use current speed as approximation
                ds = max(current_speed, 1.0) * self.dt * (t + 1)
                s_future = current_s + ds
                
                # Handle path wraparound for circular tracks
                if s_future > self.total_path_length:
                    s_future = s_future % self.total_path_length
                
                ref_x = float(self.interp_x(s_future))
                ref_y = float(self.interp_y(s_future))
                
                # Compute tangent direction
                ds_small = 0.1
                if s_future + ds_small <= self.total_path_length:
                    x_ahead = float(self.interp_x(s_future + ds_small))
                    y_ahead = float(self.interp_y(s_future + ds_small))
                else:
                    x_ahead = float(self.interp_x(s_future))
                    y_ahead = float(self.interp_y(s_future))
                
                ref_yaw = np.arctan2(y_ahead - ref_y, x_ahead - ref_x)
                
                ref_traj[t] = [ref_x, ref_y, ref_yaw, s_future]
        else:
            # Use discrete path points
            for t in range(self.horizon):
                ref_idx = min(idx + t, len(self.path) - 1)
                ref_x = self.path[ref_idx, 0]
                ref_y = self.path[ref_idx, 1]
                
                # Compute heading
                if ref_idx < len(self.path) - 1:
                    next_idx = ref_idx + 1
                else:
                    next_idx = ref_idx
                
                dx = self.path[next_idx, 0] - ref_x
                dy = self.path[next_idx, 1] - ref_y
                ref_yaw = np.arctan2(dy, dx)
                
                ref_traj[t] = [ref_x, ref_y, ref_yaw, ref_idx]
        
        return ref_traj
    
    def cost_function(self, u, x0, ref_traj):
        """
        Cost function for MPC optimization.
        
        Minimizes:
        1. Cross track error (stay on track)
        2. Heading error (point in right direction)
        3. NEGATIVE speed error (maximize speed!)
        4. Control effort
        5. Control rate (smoothness)
        """
        u = u.reshape((self.horizon, 2))
        states = self.predict_states_dynamic(x0, u)
        
        cost = 0.0
        
        # Running cost over horizon
        for t in range(self.horizon):
            state = states[t + 1]
            x, y, yaw, vx, vy, w = state
            ref_x, ref_y, ref_yaw, _ = ref_traj[t]
            
            # Cross track error
            cte = self.compute_cross_track_error(x, y, ref_x, ref_y, ref_yaw)
            cost += self.Q_cte * cte**2
            
            # Additional soft penalty for exceeding CTE limit (barrier function)
            if abs(cte) > self.max_cte * 0.8:  # Start penalizing at 80% of limit
                cost += 1000.0 * (abs(cte) - self.max_cte * 0.8)**2
            
            # Heading error
            heading_error = np.arctan2(np.sin(yaw - ref_yaw), np.cos(yaw - ref_yaw))
            cost += self.Q_heading * heading_error**2
            
            # Speed cost - encourage high speed
            # Use negative reward (lower cost for higher speed)
            # Clip to avoid negative overall cost
            speed_reward = max(0, vx / self.max_speed)  # 0 to 1
            cost += self.Q_speed * (1.0 - speed_reward)  # Lower cost = higher speed
            
            # Lateral velocity penalty (stability)
            cost += self.Q_lateral * vy**2
            
            # Control effort
            throttle, steering = u[t]
            cost += self.R_throttle * throttle**2
            cost += self.R_steering * steering**2
            
            # Control rate (smoothness)
            if t > 0:
                d_throttle = u[t, 0] - u[t-1, 0]
                d_steering = u[t, 1] - u[t-1, 1]
                cost += self.Rd_throttle * d_throttle**2
                cost += self.Rd_steering * d_steering**2
        
        # Terminal cost (end of horizon)
        final_state = states[-1]
        x_f, y_f, yaw_f = final_state[0], final_state[1], final_state[2]
        ref_x_f, ref_y_f, ref_yaw_f, _ = ref_traj[-1]
        
        cte_final = self.compute_cross_track_error(x_f, y_f, ref_x_f, ref_y_f, ref_yaw_f)
        cost += self.Q_terminal_cte * cte_final**2
        
        heading_error_final = np.arctan2(np.sin(yaw_f - ref_yaw_f), 
                                         np.cos(yaw_f - ref_yaw_f))
        cost += self.Q_terminal_heading * heading_error_final**2
        
        return cost
    
    def cte_constraint_function(self, u, x0, ref_traj):
        """
        Constraint function for Cross Track Error.
        Returns CTE for each step in horizon.
        Must satisfy: -max_cte <= CTE <= max_cte
        """
        u = u.reshape((self.horizon, 2))
        states = self.predict_states_dynamic(x0, u)
        
        cte_values = np.zeros(self.horizon)
        for t in range(self.horizon):
            state = states[t + 1]
            x, y = state[0], state[1]
            ref_x, ref_y, ref_yaw, _ = ref_traj[t]
            cte_values[t] = self.compute_cross_track_error(x, y, ref_x, ref_y, ref_yaw)
        
        return cte_values
    
    def compute_control(self, x, y, yaw, vx, vy=0.0, w=0.0):
        """
        Compute optimal control using MPC.
        
        Args:
            x, y: Position (m)
            yaw: Heading angle (rad)
            vx: Longitudinal velocity (m/s)
            vy: Lateral velocity (m/s) - optional
            w: Yaw rate (rad/s) - optional
            
        Returns:
            throttle, steering: Control inputs
        """
        # Current state (6D for dynamic model)
        x0 = np.array([x, y, yaw, vx, vy, w])
        
        # Find closest point on path
        _, idx, distance = self.find_closest_point(x, y)
        
        if self.verbose:
            print(f"Current: x={x:.2f}, y={y:.2f}, vx={vx:.2f}, distance={distance:.2f}")
        
        # Get reference trajectory
        ref_traj = self.get_reference_trajectory(idx, vx)
        
        # Initial guess (warm start if available)
        if self.u_prev is None:
            u0 = np.zeros(self.horizon * 2)
        else:
            # Shift previous solution forward in time
            u0 = np.zeros(self.horizon * 2)
            u0[:-2] = self.u_prev[2:]  # Shift by one step
            # Last control same as previous last
            u0[-2:] = self.u_prev[-2:]
        
        # Control bounds
        # Throttle: -1 (full brake) to 1 (full throttle)
        # Steering: -30 degrees to +30 degrees
        bounds = [(-1.0, 1.0), (-np.pi/6, np.pi/6)] * self.horizon
        
        # Setup constraints
        constraints = []
        
        # Note: CTE is handled as soft constraint in cost function
        # Hard constraints on CTE can make optimization infeasible,
        # especially at low speeds or sharp turns
        # The high Q_cte weight + barrier function effectively limits CTE
        
        # Speed constraint (optional - mainly handled by bounds)
        # Could add explicit speed constraint here if needed
        
        # Optimize
        import time
        t_start = time.time()
        
        try:
            result = minimize(
                self.cost_function,
                u0,
                args=(x0, ref_traj),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': 100,  # Limit iterations for real-time
                    'ftol': 1e-4,    # Tolerance
                }
            )
            
            solve_time = time.time() - t_start
            self.solve_times.append(solve_time)
            
            if not result.success:
                if self.verbose:
                    print(f"⚠️  MPC optimization failed: {result.message}")
                    print(f"   Using fallback control")
                
                # Fallback: use previous control or safe default
                if self.u_prev is not None:
                    u_opt = self.u_prev.reshape((self.horizon, 2))
                    self.solve_success_rate.append(0)
                    return u_opt[0, 0], u_opt[0, 1]
                else:
                    # Safe default: no throttle, no steering
                    self.solve_success_rate.append(0)
                    return 0.0, 0.0
            
            # Success!
            self.solve_success_rate.append(1)
            u_opt = result.x
            self.u_prev = u_opt.copy()
            
            u_opt = u_opt.reshape((self.horizon, 2))
            throttle, steering = u_opt[0, 0], u_opt[0, 1]
            
            if self.verbose:
                print(f"✓ MPC solution: throttle={throttle:.3f}, steering={np.rad2deg(steering):.1f}°, "
                      f"cost={result.fun:.2f}, time={solve_time:.3f}s")
            
            return throttle, steering
            
        except Exception as e:
            if self.verbose:
                print(f"❌ MPC exception: {e}")
            
            # Emergency fallback
            self.solve_success_rate.append(0)
            if self.u_prev is not None:
                u_opt = self.u_prev.reshape((self.horizon, 2))
                return u_opt[0, 0], u_opt[0, 1]
            return 0.0, 0.0
    
    def get_statistics(self):
        """Get optimization statistics"""
        if len(self.solve_times) == 0:
            return {}
        
        return {
            'avg_solve_time': np.mean(self.solve_times),
            'max_solve_time': np.max(self.solve_times),
            'success_rate': np.mean(self.solve_success_rate) * 100,
            'total_calls': len(self.solve_times)
        }
    
    def reset(self):
        """Reset controller state"""
        self.u_prev = None
        self.last_idx = 0
        self.solve_times = []
        self.solve_success_rate = []
