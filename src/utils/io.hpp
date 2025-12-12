#pragma once
#include <fstream>
#include <string>

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

inline void write_ghost_csv(const std::string& path, const GhostCells1D& ghosts) {
    std::ofstream f(path);
    f << "boundary,value\n";
    f << "left," << ghosts.left << "\n";
    f << "right," << ghosts.right << "\n";
}

inline void write_ghost_csv(const std::string& path, const GhostCells2D& ghosts, const Field2D& U) {
    const auto& G = *U.g;
    std::ofstream f(path);
    f << "boundary,coordinate,value\n";
    for (int j = 0; j <= G.Ny; ++j) {
        f << "west," << G.y[j] << "," << ghosts.west[j] << "\n";
        f << "east," << G.y[j] << "," << ghosts.east[j] << "\n";
    }
    for (int i = 0; i <= G.Nx; ++i) {
        f << "south," << G.x[i] << "," << ghosts.south[i] << "\n";
        f << "north," << G.x[i] << "," << ghosts.north[i] << "\n";
    }
}

inline void write_ghost_csv(const std::string& path, const GhostCells3D& ghosts, const Field3D& U) {
    const auto& G = *U.g;
    auto idx2 = [&](int a, int b, int na) { return b * (na + 1) + a; };
    std::ofstream f(path);
    f << "boundary,i,j,value\n";
    for (int j = 0; j <= G.Ny; ++j)
        for (int k = 0; k <= G.Nz; ++k) {
            f << "west," << G.y[j] << ";" << G.z[k] << "," << ghosts.west[idx2(j, k, G.Ny)] << "\n";
            f << "east," << G.y[j] << ";" << G.z[k] << "," << ghosts.east[idx2(j, k, G.Ny)] << "\n";
        }
    for (int i = 0; i <= G.Nx; ++i)
        for (int k = 0; k <= G.Nz; ++k) {
            f << "south," << G.x[i] << ";" << G.z[k] << "," << ghosts.south[idx2(i, k, G.Nx)] << "\n";
            f << "north," << G.x[i] << ";" << G.z[k] << "," << ghosts.north[idx2(i, k, G.Nx)] << "\n";
        }
    for (int i = 0; i <= G.Nx; ++i)
        for (int j = 0; j <= G.Ny; ++j) {
            f << "bottom," << G.x[i] << ";" << G.y[j] << "," << ghosts.bottom[idx2(i, j, G.Nx)] << "\n";
            f << "top," << G.x[i] << ";" << G.y[j] << "," << ghosts.top[idx2(i, j, G.Nx)] << "\n";
        }
}

inline void write_vtk(const std::string& path, const Field1D& U) {
    const auto& G = *U.g;
    std::ofstream f(path);
    f << "# vtk DataFile Version 3.0\nHeatField\nASCII\nDATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << (G.N + 1) << " 1 1\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << G.dx << " 1 1\n";
    f << "POINT_DATA " << (G.N + 1) << "\n";
    f << "SCALARS temperature double 1\nLOOKUP_TABLE default\n";
    for (double v : U.u) f << v << "\n";
}

inline void write_vtk(const std::string& path, const Field2D& U) {
    const auto& G = *U.g;
    std::ofstream f(path);
    f << "# vtk DataFile Version 3.0\nHeatField\nASCII\nDATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << (G.Nx + 1) << " " << (G.Ny + 1) << " 1\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << G.dx << " " << G.dy << " 1\n";
    f << "POINT_DATA " << (G.Nx + 1) * (G.Ny + 1) << "\n";
    f << "SCALARS temperature double 1\nLOOKUP_TABLE default\n";
    for (int j = 0; j <= G.Ny; ++j)
        for (int i = 0; i <= G.Nx; ++i)
            f << U.u[U.id(i, j)] << "\n";
}

inline void write_vtk(const std::string& path, const Field3D& U) {
    const auto& G = *U.g;
    std::ofstream f(path);
    f << "# vtk DataFile Version 3.0\nHeatField\nASCII\nDATASET STRUCTURED_POINTS\n";
    f << "DIMENSIONS " << (G.Nx + 1) << " " << (G.Ny + 1) << " " << (G.Nz + 1) << "\n";
    f << "ORIGIN 0 0 0\n";
    f << "SPACING " << G.dx << " " << G.dy << " " << G.dz << "\n";
    f << "POINT_DATA " << static_cast<size_t>(G.Nx + 1) * (G.Ny + 1) * (G.Nz + 1) << "\n";
    f << "SCALARS temperature double 1\nLOOKUP_TABLE default\n";
    for (int k = 0; k <= G.Nz; ++k)
        for (int j = 0; j <= G.Ny; ++j)
            for (int i = 0; i <= G.Nx; ++i)
                f << U.u[U.id(i, j, k)] << "\n";
}

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
