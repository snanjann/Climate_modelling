#pragma once
#include <fstream>
#include <string>
#include <vector>

#include "grid_1d.hpp"
#include "grid_2d.hpp"
#include "grid_3d.hpp"
#include "utils/data.hpp"

template <class WriteFunc>
inline void write_table_csv(const std::string& path, const std::vector<GridPointRecord>& table, WriteFunc writer) {
    std::ofstream f(path);
    f << "x,y,z,temperature,force,density,vx,vy,vz,rain\n";
    for (const auto& rec : table) writer(f, rec);
}

inline void write_table_csv(const std::string& path, const std::vector<GridPointRecord>& table) {
    write_table_csv(path, table, [](std::ofstream& f, const GridPointRecord& rec) {
        f << rec.x << "," << rec.y << "," << rec.z << "," << rec.temperature << ","
          << rec.force << "," << rec.density << "," << rec.vx << "," << rec.vy << "," << rec.vz
          << "," << rec.rain << "\n";
    });
}

void write_ghost_csv(const std::string& path, const GhostCells1D& ghosts);
void write_ghost_csv(const std::string& path, const GhostCells2D& ghosts, const Field2D& U);
void write_ghost_csv(const std::string& path, const GhostCells3D& ghosts, const Field3D& U);

void write_vtk(const std::string& path, const Field1D& U);
void write_vtk(const std::string& path, const Field2D& U);
void write_vtk(const std::string& path, const Field3D& U);

template <class FieldType, class GhostType>
inline void write_artifacts(const std::string& results_dir, const std::string& mode,
                            const SimulationResult<FieldType, GhostType>& res,
                            const std::string& vtk_prefix = "") {
    std::string table_path = results_dir + "/" + mode + "_table.csv";
    std::string ghost_path = results_dir + "/" + mode + "_ghosts.csv";
    std::string vtk_path = vtk_prefix.empty()
        ? (results_dir + "/" + mode + ".vtk")
        : (vtk_prefix + "_" + mode + ".vtk");
    write_table_csv(table_path, res.table);
    write_ghost_csv(ghost_path, res.ghosts, res.field);
    write_vtk(vtk_path, res.field);
}

// Overload for 1D ghost writer signature
inline void write_ghost_csv(const std::string& path, const GhostCells1D& ghosts, const Field1D&) {
    write_ghost_csv(path, ghosts);
}
