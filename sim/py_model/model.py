# main.py
import numpy as np
from .C16_CONTINENTAL_Tire_Data import *
from .constants import *


def sgn(v):
    return np.sign(v)


class Chassis:
    def __init__(self, X=0, Y=0, yaw=0, vx=0, vy=0, w=0):
        self.X = X
        self.Y = Y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.w = w


class ControlInfluence:
    def __init__(self, throttle=0, steeringAngle=0, brakes=0):
        self.throttle = throttle
        self.steeringAngle = steeringAngle
        self.brakes = brakes


class WheelModel:
    def __init__(self):
        self.gamma = 0.0
        self.gammay = self.gamma * LGAY
        self.Fz0 = FNOMIN
        self.Fz = m * 9.81 / 4
        self.dfz = (self.Fz - self.Fz0 * LFZO) / (self.Fz0 * LFZO)
        self.muy = (PDY1 + PDY2 * self.dfz) * (1 - PDY3 * self.gammay * self.gammay) * LMUY
        self.Cy = PCY1 * LCY
        self.Dy = self.muy * self.Fz
        self.Ky = PKY1 * self.Fz0 * np.sin(2 * np.arctan(self.Fz / (PKY2 * self.Fz0 * LFZO))) * (1 - PKY3 * np.abs(self.gammay)) * LFZO * LKY
        self.By = self.Ky / (self.Cy * self.Dy)
        self.SHy = (PHY1 + PHY2 * self.dfz) * LHY + PHY3 * self.gammay
        self.SVy = self.Fz * ((PVY1 + PVY2 * self.dfz) * LVY + (PVY3 + PVY4 * self.dfz) * self.gammay) * LMUY
        self.Cyk = RCY1
        self.Eyk = REY1 + REY2 * self.dfz
        self.SHyk = RHY1 + RHY2 * self.dfz

        self.gammax = self.gamma * LGAX
        self.mux = (PDX1 + PDX2 * self.dfz) * (1 - PDX3 * self.gammax * self.gammax) * LMUX
        self.Cx = PCX1 * LCX
        self.Dx = self.mux * self.Fz
        self.Kx = self.Fz * (PKX1 + PKX2 * self.dfz) * np.exp(PKX3 * self.dfz) * LKX
        self.Bx = self.Kx / (self.Cx * self.Dx)
        self.SHx = (PHX1 + PHX2 * self.dfz) * LHX
        self.Svx = self.Fz * (PVX1 + PVX2 * self.dfz) * LVX * LMUX
        self.Exa = REX1 + REX2 * self.dfz * self.dfz  # Note: REX2 * REX2 seems typo in original, but kept
        self.Cxa = RCX1
        self.SHxa = RHX1
        self.alpha = 0
        self.kappa = 0
        self.angleVelocity = 0

    def __mul__(self, a):
        new = type(self)()
        new.angleVelocity = self.angleVelocity * a
        return new

    def __add__(self, a):
        new = type(self)()
        new.angleVelocity = self.angleVelocity + a.angleVelocity
        return new

    def pureLateralSlip(self, alpha):
        alphay = alpha + self.SHy
        Ey = (PEY1 + PEY2 * self.dfz) * (1 - (PEY3 + PEY4 * self.gammay) * sgn(alphay)) * LEY
        Fy0 = self.Dy * np.sin(self.Cy * np.arctan(self.By * alphay - Ey * (self.By * alphay - np.arctan(self.By * alphay)))) + self.SVy
        return Fy0

    def combinedLateralSlip(self, alpha, kappa):
        Fy0 = self.pureLateralSlip(alpha)
        kappas = kappa + self.SHyk
        Byk = RBY1 * np.cos(np.arctan(RBY2 * (alpha - RBY3))) * LYKA
        Dyk = Fy0 / np.cos(self.Cyk * np.arctan(Byk * self.SHyk - self.Eyk * (Byk * self.SHyk - np.arctan(Byk * self.SHyk))))
        DVyk = self.muy * self.Fz * (RVY1 + RVY2 * self.dfz + RVY3 * self.gamma) * np.cos(np.arctan(RVY4 * alpha))
        SVyk = DVyk * np.sin(RVY5 * np.arctan(RVY6 * kappa)) * LVYKA
        Fy = Dyk * np.cos(self.Cyk * np.arctan(Byk * kappas - self.Eyk * (Byk * kappas - np.arctan(Byk * kappas)))) + SVyk
        return Fy

    def pureLongitudinalSlip(self, kappa):
        kappax = kappa + self.SHx
        Ex = (PEX1 + PEX2 * self.dfz + PEX3 * self.dfz * self.dfz) * (1 - PEX4 * sgn(kappax)) * LEX
        Fx0 = self.Dx * np.sin(self.Cx * np.arctan(self.Bx * kappax - Ex * (self.Bx * kappax - np.arctan(self.Bx * kappax)))) + self.Svx
        return Fx0

    def combinedLongitudinalSlip(self, alpha, kappa):
        Fx0 = self.pureLongitudinalSlip(kappa)
        Bxa = RBX1 * np.cos(np.arctan(RBX2 * kappa)) * LXAL
        alphas = alpha + self.SHxa
        Dxa = Fx0 / np.cos(self.Cxa * np.arctan(Bxa * self.SHxa - self.Exa * (Bxa * self.SHxa - np.arctan(Bxa * self.SHxa))))
        Fx = Dxa * np.cos(self.Cxa * np.arctan(Bxa * alphas - self.Exa * (Bxa * alphas - np.arctan(Bxa * alphas))))
        return Fx


class FrontWheel(WheelModel):
    def coeffs(self, input, actual):
        vfx = actual.vx
        vfy = actual.vy + actual.w * lf
        a = np.arctan2((actual.vy + lf * actual.w), actual.vx) - input.steeringAngle
        self.alpha = a

        ve = vfx * np.cos(input.steeringAngle) + vfy * np.sin(input.steeringAngle)
        k = (self.angleVelocity * UNLOADED_RADIUS - ve) / max(ve, vxmin)
        self.kappa = k

    def Flongitudinal(self, parentCar, ci):
        self.coeffs(ci, parentCar)
        return self.combinedLongitudinalSlip(self.alpha, self.kappa)

    def Flateral(self, parentCar, ci):
        self.coeffs(ci, parentCar)
        return self.combinedLateralSlip(self.alpha, self.kappa)

    def wheelAngleAcceleration(self, parentCar, ci):
        Fx = self.Flongitudinal(parentCar, ci)
        Frrf = Crr * np.tanh(parentCar.vx) / 2
        Fbf = ci.brakes * Cbf * np.tanh(parentCar.vx) / 2
        frontWheelMomentum = UNLOADED_RADIUS * (-Fx - Frrf - Fbf) / 2.0
        return frontWheelMomentum / Iw

    def wheelDerivate(self, parentCar, ci):
        d = FrontWheel()
        d.angleVelocity = self.wheelAngleAcceleration(parentCar, ci)
        return d


class RearWheel(WheelModel):
    def coeffs(self, input, actual):
        vrx = actual.vx
        vry = actual.vy - actual.w * lr
        a = np.arctan2((actual.vy - lr * actual.w), actual.vx)
        self.alpha = a

        vfx = actual.vx
        vfy = actual.vy + actual.w * lf
        ve = vfx * np.cos(input.steeringAngle) + vfy * np.sin(input.steeringAngle)
        k = (self.angleVelocity * UNLOADED_RADIUS - vrx) / max(vrx, vxmin)
        self.kappa = k

    def Flongitudinal(self, parentCar, ci):
        self.coeffs(ci, parentCar)
        return self.combinedLongitudinalSlip(self.alpha, self.kappa)

    def Flateral(self, parentCar, ci):
        self.coeffs(ci, parentCar)
        return self.combinedLateralSlip(self.alpha, self.kappa)

    def wheelAngleAcceleration(self, parentCar, ci):
        Fx = self.Flongitudinal(parentCar, ci)
        Fdrv = ci.throttle * Cm / 2
        Frrr = Crr * np.tanh(parentCar.vx) / 2
        Fbr = ci.brakes * Cbr * np.tanh(parentCar.vx) / 2
        frontWheelMomentum = UNLOADED_RADIUS * (Fdrv - Fx - Frrr - Fbr) / 2.0
        return frontWheelMomentum / Iw

    def wheelDerivate(self, parentCar, ci):
        d = RearWheel()
        d.angleVelocity = self.wheelAngleAcceleration(parentCar, ci)
        return d


class State:
    def __init__(self, body=None, front_left=None, front_right=None, rear_left=None, rear_right=None):
        if body is None:
            body = Chassis()
        if front_left is None:
            front_left = FrontWheel()
        if front_right is None:
            front_right = FrontWheel()
        if rear_left is None:
            rear_left = RearWheel()
        if rear_right is None:
            rear_right = RearWheel()
        self.body = body
        self.front_left = front_left
        self.front_right = front_right
        self.rear_left = rear_left
        self.rear_right = rear_right

    def __mul__(self, a):
        return State(
            Chassis(self.body.X * a, self.body.Y * a, self.body.yaw * a, self.body.vx * a, self.body.vy * a, self.body.w * a),
            self.front_left * a,
            self.front_right * a,
            self.rear_left * a,
            self.rear_right * a
        )

    def __add__(self, other):
        return State(
            Chassis(self.body.X + other.body.X, self.body.Y + other.body.Y, self.body.yaw + other.body.yaw,
                    self.body.vx + other.body.vx, self.body.vy + other.body.vy, self.body.w + other.body.w),
            self.front_left + other.front_left,
            self.front_right + other.front_right,
            self.rear_left + other.rear_left,
            self.rear_right + other.rear_right
        )


class Dynamic4WheelsModel:
    def __init__(self):
        self.carState = State()
        self.t = 0

    def Flongitudinal(self, input, actual):
        steeringAngle = input.steeringAngle

        flWheel = actual.front_left.Flongitudinal(actual.body, input) * np.cos(steeringAngle) - actual.front_left.Flateral(actual.body, input) * np.sin(steeringAngle)
        frWheel = actual.front_right.Flongitudinal(actual.body, input) * np.cos(steeringAngle) - actual.front_right.Flateral(actual.body, input) * np.sin(steeringAngle)
        rlWheel = actual.rear_left.Flongitudinal(actual.body, input)
        rrWheel = actual.rear_right.Flongitudinal(actual.body, input)

        return flWheel + frWheel + rlWheel + rrWheel

    def Flateral(self, input, actual):
        steeringAngle = input.steeringAngle

        flWheel = actual.front_left.Flongitudinal(actual.body, input) * np.sin(steeringAngle) + actual.front_left.Flateral(actual.body, input) * np.cos(steeringAngle)
        frWheel = actual.front_right.Flongitudinal(actual.body, input) * np.sin(steeringAngle) + actual.front_right.Flateral(actual.body, input) * np.cos(steeringAngle)
        rlWheel = actual.rear_left.Flateral(actual.body, input)
        rrWheel = actual.rear_right.Flateral(actual.body, input)

        return flWheel + frWheel + rlWheel + rrWheel

    def L(self, input, actual):
        steeringAngle = input.steeringAngle

        flWheelT = actual.front_left.Flongitudinal(actual.body, input) * np.cos(steeringAngle) - actual.front_left.Flateral(actual.body, input) * np.sin(steeringAngle)
        flTM = flWheelT * (-b)
        flWheelL = actual.front_left.Flongitudinal(actual.body, input) * np.sin(steeringAngle) + actual.front_left.Flateral(actual.body, input) * np.cos(steeringAngle)
        flLM = flWheelL * lf

        frWheelT = actual.front_right.Flongitudinal(actual.body, input) * np.cos(steeringAngle) - actual.front_right.Flateral(actual.body, input) * np.sin(steeringAngle)
        frTM = frWheelT * b
        frWheelL = actual.front_right.Flongitudinal(actual.body, input) * np.sin(steeringAngle) + actual.front_right.Flateral(actual.body, input) * np.cos(steeringAngle)
        frLM = frWheelL * lf

        rlWheelT = actual.rear_left.Flongitudinal(actual.body, input)
        rlTM = rlWheelT * (-b)
        rlWheelL = actual.rear_left.Flateral(actual.body, input)
        rlLM = rlWheelL * (-lr)

        rrWheelT = actual.rear_right.Flongitudinal(actual.body, input)
        rrTM = rrWheelT * b
        rrWheelL = actual.rear_right.Flateral(actual.body, input)
        rrLM = rrWheelL * (-lr)

        return flTM + flLM + frLM + frTM + rlTM + rlLM + rrTM + rrLM

    def Derivatives(self, input, actual):
        body = actual.body
        dxdt = body.vx * np.cos(body.yaw) - body.vy * np.sin(body.yaw)
        dydt = body.vx * np.sin(body.yaw) + body.vy * np.cos(body.yaw)
        dyawdt = body.w
        dvxdt = (self.Flongitudinal(input, actual) / m + body.vy * body.w)
        dvydt = (self.Flateral(input, actual) / m - body.vx * body.w)
        dwdt = self.L(input, actual) / Iz

        dFLdt = actual.front_left.wheelDerivate(body, input)
        dFRdt = actual.front_right.wheelDerivate(body, input)
        dRLdt = actual.rear_left.wheelDerivate(body, input)
        dRRdt = actual.rear_right.wheelDerivate(body, input)

        return State(Chassis(dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt), dFLdt, dFRdt, dRLdt, dRRdt)

    def getX(self):
        return self.carState.body.X

    def getY(self):
        return self.carState.body.Y

    def getyaw(self):
        return self.carState.body.yaw

    def getvx(self):
        return self.carState.body.vx

    def getvy(self):
        return self.carState.body.vy

    def getr(self):
        return self.carState.body.w

    def gett(self):
        return self.t

    def updateRK4(self, input):
        h = dt
        k1 = self.Derivatives(input, self.carState)
        k2 = self.Derivatives(input, self.carState + (k1 * (h / 2)))
        k3 = self.Derivatives(input, self.carState + (k2 * (h / 2)))
        k4 = self.Derivatives(input, self.carState + (k3 * h))

        self.carState = self.carState + ((k1 + (k2 * 2) + (k3 * 2) + k4) * (h / 6))
        self.t += dt
        
    def __repr__(self):
        state_string = f'{round(self.gett(), 3)} (x={round(self.getX(), 3)}, y={round(self.getY(), 3)}, yaw={round(self.getyaw(), 3)}), '
        velocities_string = f'(vx={round(self.getvx(), 3)}, vy={round(self.getvy(), 3)}, omega={round(self.getr(), 3)})'
        return state_string + velocities_string


def print_state(sdbm):
    print(f"x({sdbm.gett()}) = {{{sdbm.getX()} {sdbm.getY()} {sdbm.getyaw()} {sdbm.getvx()} {sdbm.getvy()} {sdbm.getr()} }}")


if __name__ == "__main__":
    iterations_by_one_step = 50
    A = Dynamic4WheelsModel()
    with open("input.txt", "r") as f:
        n = int(f.readline().strip())
        for i in range(n):
            a, sa, br = map(float, f.readline().strip().split())
            for j in range(iterations_by_one_step):
                A.updateRK4(ControlInfluence(a, sa, br))
            print_state(A)