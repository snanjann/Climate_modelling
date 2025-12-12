#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "grid_1d.hpp"
#include "grid_2d.hpp"
#include "grid_3d.hpp"

inline std::vector<double> load_geospatial_csv(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Failed to open geospatial data file: " << path << "\n";
        return {};
    }
    std::vector<double> values;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            try {
                double val = std::stod(cell);
                if (std::isfinite(val)) values.push_back(val);
            } catch (...) {
                // Skip non-numeric tokens (e.g., headers)
            }
        }
    }
    std::cout << "Loaded " << values.size() << " values from " << path << "\n";
    return values;
}

inline void clamp_values(std::vector<double>& data, double lo, double hi) {
    for (double& v : data) {
        if (!std::isfinite(v)) v = lo;
        v = std::clamp(v, lo, hi);
    }
}

template <class FieldType>
inline void clean_field(FieldType& field, double lo, double hi) {
    clamp_values(field.u, lo, hi);
}

inline bool apply_geospatial(Field1D& U, const std::vector<double>& data) {
    if (data.empty()) return false;
    if (data.size() != U.u.size()) {
        std::cout << "Geospatial data size mismatch for 1D grid (got " << data.size()
                  << ", expected " << U.u.size() << ")\n";
        return false;
    }
    U.u = data;
    return true;
}

inline bool apply_geospatial(Field2D& U, const std::vector<double>& data) {
    if (data.empty()) return false;
    if (data.size() != U.u.size()) {
        std::cout << "Geospatial data size mismatch for 2D grid (got " << data.size()
                  << ", expected " << U.u.size() << ")\n";
        return false;
    }
    U.u = data;
    return true;
}

inline bool apply_geospatial(Field3D& U, const std::vector<double>& data) {
    if (data.empty()) return false;
    if (data.size() != U.u.size()) {
        std::cout << "Geospatial data size mismatch for 3D grid (got " << data.size()
                  << ", expected " << U.u.size() << ")\n";
        return false;
    }
    U.u = data;
    return true;
}

inline std::function<double(double, double)> build_initializer(
    const Grid2D& G,
    const std::vector<double>& data,
    std::function<double(double, double)> fallback) {
    if (data.size() != static_cast<size_t>((G.Nx + 1) * (G.Ny + 1))) return fallback;
    return [&, fallback](double x, double y) {
        int i = static_cast<int>(std::lround(x / G.dx));
        int j = static_cast<int>(std::lround(y / G.dy));
        i = std::clamp(i, 0, G.Nx);
        j = std::clamp(j, 0, G.Ny);
        return data[j * (G.Nx + 1) + i];
    };
}
