#pragma once
#include <algorithm>

constexpr double BASE_DENSITY = 1.225; // kg / m^3 (air at sea level)

double compute_density(double base_density, double rain, double temperature);
double compute_force(double density, double rain);
