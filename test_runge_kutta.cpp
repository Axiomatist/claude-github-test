#include "runge_kutta.hpp"

#include <cmath>
#include <iostream>
#include <string>

// Minimal test harness
static int g_tests_run = 0;
static int g_tests_passed = 0;

static void assert_near(double actual, double expected, double tol, const std::string& name) {
    ++g_tests_run;
    if (std::abs(actual - expected) <= tol) {
        std::cout << "[PASS] " << name << "\n";
        ++g_tests_passed;
    } else {
        std::cout << "[FAIL] " << name
                  << "  got=" << actual
                  << "  expected=" << expected
                  << "  diff=" << std::abs(actual - expected)
                  << "  tol=" << tol << "\n";
    }
}

// dy/dt = -y,  y(0) = 1  =>  y(t) = e^(-t)
static void test_exponential_decay() {
    auto f = [](double /*t*/, double y) { return -y; };
    double y = runge_kutta_integrate(f, 0.0, 1.0, 1.0, 0.1);
    assert_near(y, std::exp(-1.0), 1e-6, "exponential decay  y(1) = e^-1");
}

// dy/dt = y,  y(0) = 1  =>  y(t) = e^t
static void test_exponential_growth() {
    auto f = [](double /*t*/, double y) { return y; };
    double y = runge_kutta_integrate(f, 0.0, 1.0, 1.0, 0.1);
    assert_near(y, std::exp(1.0), 1e-6, "exponential growth  y(1) = e^1");
}

// dy/dt = 0,  y(0) = 5  =>  y(t) = 5  (constant stays constant)
static void test_constant_function() {
    auto f = [](double /*t*/, double /*y*/) { return 0.0; };
    double y = runge_kutta_integrate(f, 0.0, 5.0, 3.0, 0.5);
    assert_near(y, 5.0, 1e-14, "constant: dy/dt = 0, y(3) = 5");
}

// dy/dt = 2t,  y(0) = 0  =>  y(t) = t^2  (exact for RK4)
static void test_quadratic() {
    auto f = [](double t, double /*y*/) { return 2.0 * t; };
    double y = runge_kutta_integrate(f, 0.0, 0.0, 3.0, 0.5);
    assert_near(y, 9.0, 1e-12, "quadratic: dy/dt = 2t, y(3) = 9");
}

// dy/dt = 3t^2,  y(0) = 0  =>  y(t) = t^3
static void test_cubic() {
    auto f = [](double t, double /*y*/) { return 3.0 * t * t; };
    double y = runge_kutta_integrate(f, 0.0, 0.0, 2.0, 0.1);
    assert_near(y, 8.0, 1e-10, "cubic: dy/dt = 3t^2, y(2) = 8");
}

// Single-step accuracy: dy/dt = -y, y(0) = 1, h = 0.1
// RK4 local error is O(h^5); verify the step is within 1e-7
static void test_single_step_accuracy() {
    auto f = [](double /*t*/, double y) { return -y; };
    double y = runge_kutta_step(f, 0.0, 1.0, 0.1);
    assert_near(y, std::exp(-0.1), 1e-7, "single step accuracy (h=0.1)");
}

// dy/dt = cos(t),  y(0) = 0  =>  y(t) = sin(t)
static void test_sinusoidal() {
    auto f = [](double t, double /*y*/) { return std::cos(t); };
    double y = runge_kutta_integrate(f, 0.0, 0.0, M_PI / 2.0, 0.05);
    assert_near(y, 1.0, 1e-8, "sinusoidal: dy/dt = cos(t), y(pi/2) = 1");
}

int main() {
    test_exponential_decay();
    test_exponential_growth();
    test_constant_function();
    test_quadratic();
    test_cubic();
    test_single_step_accuracy();
    test_sinusoidal();

    std::cout << "\n" << g_tests_passed << "/" << g_tests_run << " tests passed.\n";
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}
