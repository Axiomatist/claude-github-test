#pragma once

#include <algorithm>

// Performs a single step of the 4th-order Runge-Kutta method.
//
// Solves dy/dt = f(t, y) from t to t + h given initial value y.
//
// Parameters:
//   f  - callable with signature double(double t, double y)
//   t  - current time
//   y  - current state
//   h  - step size
//
// Returns the state at t + h.
template <typename Callable>
double runge_kutta_step(Callable f, double t, double y, double h) {
    double k1 = f(t,           y);
    double k2 = f(t + h / 2.0, y + (h / 2.0) * k1);
    double k3 = f(t + h / 2.0, y + (h / 2.0) * k2);
    double k4 = f(t + h,       y + h * k3);
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// Integrates dy/dt = f(t, y) from t0 to t_end using the 4th-order Runge-Kutta
// method with step size h.
//
// Parameters:
//   f      - callable with signature double(double t, double y)
//   t0     - initial time
//   y0     - initial state at t0
//   t_end  - end time
//   h      - step size (the last step is shortened to reach t_end exactly)
//
// Returns the state at t_end.
template <typename Callable>
double runge_kutta_integrate(Callable f, double t0, double y0, double t_end, double h) {
    double t = t0;
    double y = y0;
    while (t < t_end - 1e-12) {
        double step = std::min(h, t_end - t);
        y = runge_kutta_step(f, t, y, step);
        t += step;
    }
    return y;
}
