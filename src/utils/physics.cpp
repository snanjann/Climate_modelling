#include "physics.hpp"
#include <algorithm>

double compute_density(double base_density, double rain, double temperature) {
    double rho = base_density * (1.0 + 0.02 * rain) - 0.001 * (temperature - 293.15);
    return std::clamp(rho, 0.2 * base_density, 5.0 * base_density);
}

double compute_force(double density, double rain) {
    return density * rain * 9.81;
}
