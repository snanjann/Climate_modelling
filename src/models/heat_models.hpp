#pragma once
#include <vector>

#include "grid_2d.hpp"
#include "grid_3d.hpp"
#include "utils/data.hpp"
#include "utils/weather.hpp"

// Legacy/simple modes used for visualization.
Field2D run_diffusion_2d_simple();
Field2D run_advection_2d_simple();
Field2D run_forcing_2d_simple();
Field3D run_full_climate_3d_simple();
