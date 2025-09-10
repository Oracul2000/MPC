import numpy as np
from collections import deque
import time

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(None, None), integrator_limits=(None, None)):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.min_out, self.max_out = output_limits
        self.min_i, self.max_i = integrator_limits

        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def reset(self, integral=0.0):
        self.integral = integral
        self.prev_error = None
        self.prev_time = None

    def compute(self, error, dt=None, current_time=None):
        # dt resolution
        if dt is None:
            if current_time is None or self.prev_time is None:
                dt = 0.0
            else:
                dt = current_time - self.prev_time
        # Derivative
        derivative = 0.0
        if self.prev_error is not None and dt > 0.0:
            derivative = (error - self.prev_error) / dt
        # Integral with simple trapezoidal approximation
        if dt > 0.0:
            self.integral += 0.5 * (error + (self.prev_error if self.prev_error is not None else error)) * dt

        # anti-windup: clamp integrator
        if self.min_i is not None:
            self.integral = max(self.min_i, self.integral)
        if self.max_i is not None:
            self.integral = min(self.max_i, self.integral)

        out = self.kp * error + self.ki * self.integral + self.kd * derivative

        # saturate output
        if self.min_out is not None:
            out = max(self.min_out, out)
        if self.max_out is not None:
            out = min(self.max_out, out)

        # save state
        self.prev_error = error
        if current_time is not None:
            self.prev_time = current_time

        return out
    
    
class LongitudinalPID:
    def __init__(self, kp, ki, kd, v_max=None, throttle_limits=(-1.0, 1.0), integrator_limits=(None, None)):
        self.pid = PIDController(kp, ki, kd, output_limits=throttle_limits, integrator_limits=integrator_limits)
        self.v_max = v_max

    def reset(self):
        self.pid.reset()

    def compute(self, v_ref, v_curr, dt=None, current_time=None):
        error = v_ref - v_curr
        out = self.pid.compute(error, dt=dt, current_time=current_time)
        return out  # in throttle units -1..1 (user decides mapping to actuator)
