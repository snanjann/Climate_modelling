#pragma once
#include <functional>
#include <string>
#include <vector>

#include "grid_1d.hpp"
#include "grid_2d.hpp"
#include "grid_3d.hpp"

std::vector<double> load_geospatial_csv(const std::string& path);
void clamp_values(std::vector<double>& data, double lo, double hi);

template <class FieldType>
inline void clean_field(FieldType& field, double lo, double hi) {
    clamp_values(field.u, lo, hi);
}

bool apply_geospatial(Field1D& U, const std::vector<double>& data);
bool apply_geospatial(Field2D& U, const std::vector<double>& data);
bool apply_geospatial(Field3D& U, const std::vector<double>& data);

std::function<double(double, double)> build_initializer(
    const Grid2D& G,
    const std::vector<double>& data,
    std::function<double(double, double)> fallback);
