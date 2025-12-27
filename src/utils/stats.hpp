#pragma once
#include <string>

#include "grid_1d.hpp"
#include "grid_2d.hpp"
#include "grid_3d.hpp"

void report_summary(const std::string& mode, const Field1D& U);
void report_summary(const std::string& mode, const Field2D& U);
void report_summary(const std::string& mode, const Field3D& U);
