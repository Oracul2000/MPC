class PID:
    def __init__(self, kp, ki, kd, target=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0
        self.prev_error = 0

    def update(self, current, dt):
        error = self.target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output