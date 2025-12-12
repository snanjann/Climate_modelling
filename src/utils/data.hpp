#pragma once
#include <string>
#include <vector>

struct GridPointRecord {
    double x, y, z;
    double temperature;
    double force;
    double density;
    double vx, vy, vz;
    double rain;
};

struct GhostCells1D {
    double left;
    double right;
};

struct GhostCells2D {
    std::vector<double> west, east, south, north;
};

struct GhostCells3D {
    std::vector<double> west, east, south, north, bottom, top;
};

template <class FieldType, class GhostType>
struct SimulationResult {
    FieldType field;
    GhostType ghosts;
    std::vector<GridPointRecord> table;
    double final_time;
};
