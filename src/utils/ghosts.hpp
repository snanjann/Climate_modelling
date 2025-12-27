#pragma once
#include <vector>
#include "grid_1d.hpp"
#include "grid_2d.hpp"
#include "grid_3d.hpp"
#include "utils/data.hpp"

GhostCells1D capture_ghost_cells(const Field1D& U);
GhostCells2D capture_ghost_cells(const Field2D& U);
GhostCells3D capture_ghost_cells(const Field3D& U);
