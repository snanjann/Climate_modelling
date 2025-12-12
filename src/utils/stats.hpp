#pragma once
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "grid_1d.hpp"
#include "grid_2d.hpp"
#include "grid_3d.hpp"

inline void report_summary(const std::string& mode, const Field1D& U) {
    auto [mn_it, mx_it] = std::minmax_element(U.u.begin(), U.u.end());
    std::cout << mode << " -> min: " << *mn_it << ", max: " << *mx_it
              << ", avg: " << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << "\n";
}

inline void report_summary(const std::string& mode, const Field2D& U) {
    auto [mn_it, mx_it] = std::minmax_element(U.u.begin(), U.u.end());
    std::cout << mode << " -> min: " << *mn_it << ", max: " << *mx_it
              << ", avg: " << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << "\n";
}

inline void report_summary(const std::string& mode, const Field3D& U) {
    auto [mn_it, mx_it] = std::minmax_element(U.u.begin(), U.u.end());
    std::cout << mode << " -> min: " << *mn_it << ", max: " << *mx_it
              << ", avg: " << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << "\n";
}
