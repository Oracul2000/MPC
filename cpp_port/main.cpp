#include <cmath>
#include <algorithm> // Для std::max
#include "C16_CONTINENTAL_Tire_Data.h"
#include "constants.h"

// Используем std::max из <algorithm>
using std::max;

// Пространства имен для констант
using namespace C16_CONTINENTAL_Tire_Data;
using namespace CONSTANTS_JBS;

// Вспомогательная функция
float sgn(float v) {
	return (v > 0) - (v < 0);
}

// Структуры данных
struct chassis {
	double X, Y, yaw, vx, vy, w;
};

struct controlInfluence {
	double throttle, steeringAngle, brakes;
};


class WheelModel {
public:
	double gamma, gammay, Fz0, dfz, muy, Fz, Cy, Dy, Ky, By, SHy, SVy, Cyk, Eyk, SHyk;
	double gammax, mux, Cx, Dx, Kx, Bx, SHx, Svx, Exa, Cxa, SHxa;
	double alpha, kappa;
    double angleVelocity;

	WheelModel() {
        angleVelocity = 0.0;
		gamma = 0.;
		gammay = gamma * LGAY;
		Fz0 = FNOMIN;
		Fz = m * 9.81 / 4;
		dfz = (Fz - Fz0 * LFZO) / (Fz0 * LFZO);
		muy = (PDY1 + PDY2 * dfz) * (1 - PDY3 * gammay * gammay) * LMUY;
		Cy = PCY1 * LCY;
		Dy = muy * Fz;
		Ky = PKY1 * Fz0 * sin(2 * atan2(Fz, (PKY2 * Fz0 * LFZO))) * (1 - PKY3 * abs(gammay)) * LFZO * LKY;
		By = Ky / (Cy * Dy);
		SHy = (PHY1 + PHY2 * dfz) * LHY + PHY3 * gammay;
		SVy = Fz * ((PVY1 + PVY2 * dfz) * LVY + (PVY3 + PVY4 * dfz) * gammay) * LMUY;
		Cyk = RCY1;
		Eyk = REY1 + REY2 * dfz;
		SHyk = RHY1 + RHY2 * dfz;

		gammax = gamma * LGAX;
		mux = (PDX1 + PDX2 * dfz) * (1 - PDX3 * gammax * gammax) * LMUX;
		Cx = PCX1 * LCX;
		Dx = mux * Fz;
		Kx = Fz * (PKX1 + PKX2 * dfz) * exp(PKX3 * dfz) * LKX;
		Bx = Kx / (Cx * Dx);
		SHx = (PHX1 + PHX2 * dfz) * LHX;
		Svx = Fz * (PVX1 + PVX2 * dfz) * LVX * LMUX;
		Exa = REX1 + REX2 * REX2 * dfz;
		Cxa = RCX1;
		SHxa = RHX1;
	}

	double pureLateralSlip(double alpha_in) {
		float alphay = alpha_in + SHy;
		float Ey = (PEY1 + PEY2 * dfz) * (1 - (PEY3 + PEY4 * gammay) * sgn(alphay)) * LEY;
		float Fy0 = Dy * sin(Cy * atan(By * alphay - Ey * (By * alphay - atan(By * alphay)))) + SVy;
		return Fy0;
	}
	double combinedLateralSlip(double alpha_in, double kappa_in) {
		double Fy0 = pureLateralSlip(alpha_in);
		double kappas = kappa_in + SHyk;
		double Byk = RBY1 * cos(atan(RBY2 * (alpha_in - RBY3))) * LYKA;
		double Dyk = Fy0 / cos(Cyk * atan(Byk * SHyk - Eyk * (Byk * SHyk - atan(Byk * SHyk))));
		double DVyk = muy * Fz * (RVY1 + RVY2 * dfz + RVY3 * gamma) * cos(atan(RVY4 * alpha_in));
		double SVyk = DVyk * sin(RVY5 * atan(RVY6 * kappa_in)) * LVYKA;
		double Fy = Dyk * cos(Cyk * atan(Byk * kappas - Eyk * (Byk * kappas - atan(Byk * kappas)))) + SVyk;
		return Fy;
	}
	double pureLongitudinalSlip(double kappa_in) {
		double kappax = kappa_in + SHx;
		double Ex = (PEX1 + PEX2 * dfz + PEX3 * dfz * dfz) * (1 - PEX4 * sgn(kappax)) * LEX;
		double Fx0 = Dx * sin(Cx * atan(Bx * kappax - Ex * (Bx * kappax - atan(Bx * kappax)))) + Svx;
		return Fx0;
	}
	double combinedLongitudinalSlip(double alpha_in, double kappa_in) {
		double Fx0 = pureLongitudinalSlip(kappa_in);
		double Bxa = RBX1 * cos(atan(RBX2 * kappa_in)) * LXAL;
		double alphas = alpha_in + SHxa;
		double Dxa = Fx0 / cos(Cxa * atan(Bxa * SHxa - Exa * (Bxa * SHxa - atan(Bxa * SHxa))));
		double Fx = Dxa * cos(Cxa * atan(Bxa * alphas - Exa * (Bxa * alphas - atan(Bxa * alphas))));
		return Fx;
	}
};

class FrontWheel : public WheelModel {
public:
	void coeffs(const controlInfluence& input, const chassis& actual) {
		double vfy = actual.vy + actual.w * lf;
		alpha = atan2(vfy, actual.vx) - input.steeringAngle;

		double ve = actual.vx * cos(input.steeringAngle) + vfy * sin(input.steeringAngle);
		kappa = (angleVelocity * UNLOADED_RADIUS - ve) / max(ve, vxmin);
	}

	double Flongitudinal(const chassis& parentCar, const controlInfluence& ci) {
		coeffs(ci, parentCar);
		return combinedLongitudinalSlip(alpha, kappa);
	}
	double Flateral(const chassis& parentCar, const controlInfluence& ci) {
		coeffs(ci, parentCar);
		return combinedLateralSlip(alpha, kappa);
	}

	double wheelAngleAcceleration(const chassis& parentCar, const controlInfluence& ci) {
		double Fx = Flongitudinal(parentCar, ci);
		double Frrf = Crr * tanh(parentCar.vx) / 2.0;
		double Fbf = ci.brakes * Cbf * tanh(parentCar.vx) / 2.0;
		double frontWheelMomentum = UNLOADED_RADIUS * (-Fx - Frrf - Fbf) / 2.0;
		return frontWheelMomentum / Iw;
	}
};


class RearWheel : public WheelModel {
public:
	void coeffs(const controlInfluence& input, const chassis& actual) {
		double vry = actual.vy - actual.w * lr;
		alpha = atan2(vry, actual.vx);
		kappa = (angleVelocity * UNLOADED_RADIUS - actual.vx) / max(actual.vx, vxmin);
	}

	double Flongitudinal(const chassis& parentCar, const controlInfluence& ci) {
		coeffs(ci, parentCar);
		return combinedLongitudinalSlip(alpha, kappa);
	}
	double Flateral(const chassis& parentCar, const controlInfluence& ci) {
		coeffs(ci, parentCar);
		return combinedLateralSlip(alpha, kappa);
	}

	double wheelAngleAcceleration(const chassis& parentCar, const controlInfluence& ci) {
		double Fx = Flongitudinal(parentCar, ci);
		double Fdrv = ci.throttle * Cm / 2.0;
		double Frrr = Crr * tanh(parentCar.vx) / 2.0;
		double Fbr = ci.brakes * Cbr * tanh(parentCar.vx) / 2.0;
		double rearWheelMomentum = UNLOADED_RADIUS * (Fdrv - Fx - Frrr - Fbr) / 2.0;
		return rearWheelMomentum / Iw;
	}
};

struct state {
	chassis body;
	FrontWheel front_left, front_right;
	RearWheel rear_left, rear_right;
};

// Вспомогательные функции для математики состояний
state state_add(const state& a, const state& b) {
    state result;
    result.body.X = a.body.X + b.body.X;
    result.body.Y = a.body.Y + b.body.Y;
    result.body.yaw = a.body.yaw + b.body.yaw;
    result.body.vx = a.body.vx + b.body.vx;
    result.body.vy = a.body.vy + b.body.vy;
    result.body.w = a.body.w + b.body.w;
    result.front_left.angleVelocity = a.front_left.angleVelocity + b.front_left.angleVelocity;
    result.front_right.angleVelocity = a.front_right.angleVelocity + b.front_right.angleVelocity;
    result.rear_left.angleVelocity = a.rear_left.angleVelocity + b.rear_left.angleVelocity;
    result.rear_right.angleVelocity = a.rear_right.angleVelocity + b.rear_right.angleVelocity;
    return result;
}

state state_mul_scalar(const state& s, double scalar) {
    state result;
    result.body.X = s.body.X * scalar;
    result.body.Y = s.body.Y * scalar;
    result.body.yaw = s.body.yaw * scalar;
    result.body.vx = s.body.vx * scalar;
    result.body.vy = s.body.vy * scalar;
    result.body.w = s.body.w * scalar;
    result.front_left.angleVelocity = s.front_left.angleVelocity * scalar;
    result.front_right.angleVelocity = s.front_right.angleVelocity * scalar;
    result.rear_left.angleVelocity = s.rear_left.angleVelocity * scalar;
    result.rear_right.angleVelocity = s.rear_right.angleVelocity * scalar;
    return result;
}


class Dynamic4WheelsModel {
public:
	state carState;
	double t = 0;

	Dynamic4WheelsModel() {
		carState.body = { 0, 0, 0, 0, 0, 0 };
	}

    void set_initial_state(double x = 0.0, double y = 0.0, double yaw = 0.0, double vx = 0.0, double vy = 0.0, double w = 0.0) {
        carState.body.X = x;
        carState.body.Y = y;
        carState.body.yaw = yaw;
        carState.body.vx = vx;
        carState.body.vy = vy;
        carState.body.w = w;
    }

	double Flongitudinal(const controlInfluence& input, state& actual) {
		double steeringAngle = input.steeringAngle;
		double flWheel = actual.front_left.Flongitudinal(actual.body, input) * cos(steeringAngle) - actual.front_left.Flateral(actual.body, input) * sin(steeringAngle);
		double frWheel = actual.front_right.Flongitudinal(actual.body, input) * cos(steeringAngle) - actual.front_right.Flateral(actual.body, input) * sin(steeringAngle);
		double rlWheel = actual.rear_left.Flongitudinal(actual.body, input);
		double rrWheel = actual.rear_right.Flongitudinal(actual.body, input);
		return flWheel + frWheel + rlWheel + rrWheel;
	}

	double Flateral(const controlInfluence& input, state& actual) {
		double steeringAngle = input.steeringAngle;
		double flWheel = actual.front_left.Flongitudinal(actual.body, input) * sin(steeringAngle) + actual.front_left.Flateral(actual.body, input) * cos(steeringAngle);
		double frWheel = actual.front_right.Flongitudinal(actual.body, input) * sin(steeringAngle) + actual.front_right.Flateral(actual.body, input) * cos(steeringAngle);
		double rlWheel = actual.rear_left.Flateral(actual.body, input);
		double rrWheel = actual.rear_right.Flateral(actual.body, input);
		return flWheel + frWheel + rlWheel + rrWheel;
	}

	double L(const controlInfluence& input, state& actual) {
        double steer = input.steeringAngle;
		double fl_lat = actual.front_left.Flateral(actual.body, input);
        double fr_lat = actual.front_right.Flateral(actual.body, input);
        double rl_lat = actual.rear_left.Flateral(actual.body, input);
        double rr_lat = actual.rear_right.Flateral(actual.body, input);

        double fl_lon = actual.front_left.Flongitudinal(actual.body, input);
        double fr_lon = actual.front_right.Flongitudinal(actual.body, input);
        double rl_lon = actual.rear_left.Flongitudinal(actual.body, input);
        double rr_lon = actual.rear_right.Flongitudinal(actual.body, input);

        double Mz = lf * (fl_lat * cos(steer) + fl_lon * sin(steer) + fr_lat * cos(steer) + fr_lon * sin(steer))
                  - lr * (rl_lat + rr_lat)
                  + b * (-fl_lon * cos(steer) + fl_lat * sin(steer) + fr_lon * cos(steer) - fr_lat * sin(steer)
                         - rl_lon + rr_lon);
        return Mz;
	}

	state Derivatives(const controlInfluence& input, state& actual) {
		chassis& body = actual.body;
		double dxdt = body.vx * cos(body.yaw) - body.vy * sin(body.yaw);
		double dydt = body.vx * sin(body.yaw) + body.vy * cos(body.yaw);
		double dyawdt = body.w;
		double dvxdt = Flongitudinal(input, actual) / m + body.vy * body.w;
		double dvydt = Flateral(input, actual) / m - body.vx * body.w;
		double dwdt = L(input, actual) / Iz;

        state d_state;
        d_state.body = {dxdt, dydt, dyawdt, dvxdt, dvydt, dwdt};
        d_state.front_left.angleVelocity = actual.front_left.wheelAngleAcceleration(body, input);
        d_state.front_right.angleVelocity = actual.front_right.wheelAngleAcceleration(body, input);
        d_state.rear_left.angleVelocity = actual.rear_left.wheelAngleAcceleration(body, input);
        d_state.rear_right.angleVelocity = actual.rear_right.wheelAngleAcceleration(body, input);
        
		return d_state;
	}
    
    // Метод получения текущего состояния шасси
    chassis getChassisState() const { return carState.body; }

	void updateRK4(const controlInfluence& input) {
		float h = dt;
        state k1 = Derivatives(input, carState);
        state temp2 = state_add(carState, state_mul_scalar(k1, h / 2.0));
        state k2 = Derivatives(input, temp2);
        state temp3 = state_add(carState, state_mul_scalar(k2, h / 2.0));
        state k3 = Derivatives(input, temp3);
        state temp4 = state_add(carState, state_mul_scalar(k3, h));
        state k4 = Derivatives(input, temp4);

        state weighted_sum = state_add(k1, state_mul_scalar(k2, 2.0));
        weighted_sum = state_add(weighted_sum, state_mul_scalar(k3, 2.0));
        weighted_sum = state_add(weighted_sum, k4);
        
        carState = state_add(carState, state_mul_scalar(weighted_sum, h / 6.0));
		t += dt;
	}
};