// Heat conduction and convection workflows in 1D/2D/3D with data prep + visualization.
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

#include "src/common.hpp"
#include "src/grid_1d.hpp"
#include "src/grid_2d.hpp"
#include "src/grid_3d.hpp"

#include "src/operators/laplace_1d.hpp"
#include "src/operators/laplace_2d.hpp"
#include "src/operators/laplace_3d.hpp"
#include "src/operators/advection.hpp"

#include "src/solvers/heat_explicit.hpp"
#include "src/solvers/heat_adi.hpp"
#include "src/solvers/heat_3d_explicit.hpp"
#include "src/solvers/advection_diffusion.hpp"

namespace fs = std::filesystem;

const std::string RESULTS_DIR = "Results_updated";
constexpr double BASE_DENSITY = 1.225; // kg / m^3 (air at sea level)

struct CLIOptions {
    std::string mode;
    std::string geodata_path;
    std::string vtk_prefix;
};

struct WeatherModel {
    double ambient_temp;
    std::function<double(double, double, double, double)> rain_rate;
    std::function<std::array<double, 3>(double, double, double, double)> wind_field;
};

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

CLIOptions parse_cli(int argc, char** argv) {
    CLIOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            opts.mode = argv[++i];
        } else if (arg == "--geodata" && i + 1 < argc) {
            opts.geodata_path = argv[++i];
        } else if (arg == "--vtk" && i + 1 < argc) {
            opts.vtk_prefix = argv[++i];
        } else {
            std::cout << "Unknown or incomplete argument: " << arg << "\n";
            std::cout << "Usage: ./solver --mode <mode> [--geodata file.csv] [--vtk prefix]\n";
            std::exit(1);
        }
    }
    if (opts.mode.empty()) {
        std::cout << "Usage: ./solver --mode <mode> [--geodata file.csv] [--vtk prefix]\n";
        std::exit(1);
    }
    return opts;
}

WeatherModel default_weather_model() {
    WeatherModel weather;
    weather.ambient_temp = 293.15; // 20 C
    weather.rain_rate = [](double x, double y, double z, double t) {
        return 0.01 + 0.005 * std::sin(2.0 * M_PI * x + 0.5 * t)
               + 0.002 * std::cos(2.0 * M_PI * y + z + 0.1 * t);
    };
    weather.wind_field = [](double x, double y, double z, double t) {
        double u = 0.4 + 0.1 * std::cos(2.0 * M_PI * y + 0.2 * t);
        double v = 0.2 * std::sin(2.0 * M_PI * x + 0.3 * t);
        double w = 0.1 * std::cos(2.0 * M_PI * z - 0.1 * t);
        return std::array<double, 3>{u, v, w};
    };
    return weather;
}

std::vector<double> load_geospatial_csv(const std::string& path) {
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

void clamp_values(std::vector<double>& data, double lo, double hi) {
    for (double& v : data) {
        if (!std::isfinite(v)) v = lo;
        v = std::clamp(v, lo, hi);
    }
}

template <class FieldType>
void clean_field(FieldType& field, double lo, double hi) {
    clamp_values(field.u, lo, hi);
}

GhostCells1D capture_ghost_cells(const Field1D& U) {
    return {U.u.front(), U.u.back()};
}

GhostCells2D capture_ghost_cells(const Field2D& U) {
    const auto& G = *U.g;
    GhostCells2D ghosts;
    ghosts.west.resize(G.Ny + 1);
    ghosts.east.resize(G.Ny + 1);
    ghosts.south.resize(G.Nx + 1);
    ghosts.north.resize(G.Nx + 1);
    for (int j = 0; j <= G.Ny; ++j) {
        ghosts.west[j] = U.u[U.id(0, j)];
        ghosts.east[j] = U.u[U.id(G.Nx, j)];
    }
    for (int i = 0; i <= G.Nx; ++i) {
        ghosts.south[i] = U.u[U.id(i, 0)];
        ghosts.north[i] = U.u[U.id(i, G.Ny)];
    }
    return ghosts;
}

GhostCells3D capture_ghost_cells(const Field3D& U) {
    const auto& G = *U.g;
    GhostCells3D ghosts;
    ghosts.west.resize((G.Ny + 1) * (G.Nz + 1));
    ghosts.east = ghosts.west;
    ghosts.south.resize((G.Nx + 1) * (G.Nz + 1));
    ghosts.north = ghosts.south;
    ghosts.bottom.resize((G.Nx + 1) * (G.Ny + 1));
    ghosts.top = ghosts.bottom;

    auto idx2 = [&](int a, int b, int na) { return b * (na + 1) + a; };

    for (int j = 0; j <= G.Ny; ++j)
        for (int k = 0; k <= G.Nz; ++k) {
            ghosts.west[idx2(j, k, G.Ny)] = U.u[U.id(0, j, k)];
            ghosts.east[idx2(j, k, G.Ny)] = U.u[U.id(G.Nx, j, k)];
        }
    for (int i = 0; i <= G.Nx; ++i)
        for (int k = 0; k <= G.Nz; ++k) {
            ghosts.south[idx2(i, k, G.Nx)] = U.u[U.id(i, 0, k)];
            ghosts.north[idx2(i, k, G.Nx)] = U.u[U.id(i, G.Ny, k)];
        }
    for (int i = 0; i <= G.Nx; ++i)
        for (int j = 0; j <= G.Ny; ++j) {
            ghosts.bottom[idx2(i, j, G.Nx)] = U.u[U.id(i, j, 0)];
            ghosts.top[idx2(i, j, G.Nx)] = U.u[U.id(i, j, G.Nz)];
        }
    return ghosts;
}

double compute_density(double base_density, double rain, double temperature) {
    double rho = base_density * (1.0 + 0.02 * rain) - 0.001 * (temperature - 293.15);
    return std::clamp(rho, 0.2 * base_density, 5.0 * base_density);
}

double compute_force(double density, double rain) {
    return density * rain * 9.81;
}

template <class WriteFunc>
void write_table_csv(const std::string& path, const std::vector<GridPointRecord>& table, WriteFunc writer) {
    std::ofstream f(path);
    f << "x,y,z,temperature,force,density,vx,vy,vz,rain\n";
    for (const auto& rec : table) writer(f, rec);
}

void write_table_csv(const std::string& path, const std::vector<GridPointRecord>& table) {
    write_table_csv(path, table, [](std::ofstream& f, const GridPointRecord& rec) {
        f << rec.x << "," << rec.y << "," << rec.z << "," << rec.temperature << ","
          << rec.force << "," << rec.density << "," << rec.vx << "," << rec.vy << "," << rec.vz
          << "," << rec.rain << "\n";
    });
}

void write_ghost_csv(const std::string& path, const GhostCells1D& ghosts) {
    std::ofstream f(path);
    f << "boundary,value\n";
    f << "left," << ghosts.left << "\n";
    f << "right," << ghosts.right << "\n";
}

void write_ghost_csv(const std::string& path, const GhostCells2D& ghosts, const Field2D& U) {
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

void write_ghost_csv(const std::string& path, const GhostCells3D& ghosts, const Field3D& U) {
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

void write_vtk(const std::string& path, const Field1D& U) {
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

void write_vtk(const std::string& path, const Field2D& U) {
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

void write_vtk(const std::string& path, const Field3D& U) {
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

std::vector<GridPointRecord> make_table(const Field1D& U, const WeatherModel& weather, double time) {
    const auto& G = *U.g;
    std::vector<GridPointRecord> table;
    table.reserve(G.N + 1);
    for (int i = 0; i <= G.N; ++i) {
        double x = G.x[i];
        auto wind = weather.wind_field(x, 0.0, 0.0, time);
        double rain = weather.rain_rate(x, 0.0, 0.0, time);
        double density = compute_density(BASE_DENSITY, rain, U.u[i]);
        double force = compute_force(density, rain);
        table.push_back({x, 0.0, 0.0, U.u[i], force, density, wind[0], 0.0, 0.0, rain});
    }
    return table;
}

std::vector<GridPointRecord> make_table(const Field2D& U, const WeatherModel& weather, double time) {
    const auto& G = *U.g;
    std::vector<GridPointRecord> table;
    table.reserve(static_cast<size_t>(G.Nx + 1) * (G.Ny + 1));
    for (int j = 0; j <= G.Ny; ++j)
        for (int i = 0; i <= G.Nx; ++i) {
            double x = G.x[i], y = G.y[j];
            auto wind = weather.wind_field(x, y, 0.0, time);
            double rain = weather.rain_rate(x, y, 0.0, time);
            double T = U.u[U.id(i, j)];
            double density = compute_density(BASE_DENSITY, rain, T);
            double force = compute_force(density, rain);
            table.push_back({x, y, 0.0, T, force, density, wind[0], wind[1], 0.0, rain});
        }
    return table;
}

std::vector<GridPointRecord> make_table(const Field3D& U, const WeatherModel& weather, double time) {
    const auto& G = *U.g;
    std::vector<GridPointRecord> table;
    table.reserve(static_cast<size_t>(G.Nx + 1) * (G.Ny + 1) * (G.Nz + 1));
    for (int k = 0; k <= G.Nz; ++k)
        for (int j = 0; j <= G.Ny; ++j)
            for (int i = 0; i <= G.Nx; ++i) {
                double x = G.x[i], y = G.y[j], z = G.z[k];
                auto wind = weather.wind_field(x, y, z, time);
                double rain = weather.rain_rate(x, y, z, time);
                double T = U.u[U.id(i, j, k)];
                double density = compute_density(BASE_DENSITY, rain, T);
                double force = compute_force(density, rain);
                table.push_back({x, y, z, T, force, density, wind[0], wind[1], wind[2], rain});
            }
    return table;
}

bool apply_geospatial(Field1D& U, const std::vector<double>& data) {
    if (data.empty()) return false;
    if (data.size() != U.u.size()) {
        std::cout << "Geospatial data size mismatch for 1D grid (got " << data.size()
                  << ", expected " << U.u.size() << ")\n";
        return false;
    }
    U.u = data;
    return true;
}

bool apply_geospatial(Field2D& U, const std::vector<double>& data) {
    if (data.empty()) return false;
    if (data.size() != U.u.size()) {
        std::cout << "Geospatial data size mismatch for 2D grid (got " << data.size()
                  << ", expected " << U.u.size() << ")\n";
        return false;
    }
    U.u = data;
    return true;
}

bool apply_geospatial(Field3D& U, const std::vector<double>& data) {
    if (data.empty()) return false;
    if (data.size() != U.u.size()) {
        std::cout << "Geospatial data size mismatch for 3D grid (got " << data.size()
                  << ", expected " << U.u.size() << ")\n";
        return false;
    }
    U.u = data;
    return true;
}

std::function<double(double, double)> build_initializer(const Grid2D& G, const std::vector<double>& data,
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

SimulationResult<Field1D, GhostCells1D> run_heat_conduction_1d(
    const WeatherModel& weather, const std::vector<double>& geodata) {
    Grid1D G(1.0, 256);
    DirichletBC1D bc(
        [&weather](double t) { return weather.ambient_temp - 5.0 * weather.rain_rate(0.0, 0.0, 0.0, t); },
        [&weather](double t) { return weather.ambient_temp + 2.0 * std::sin(1.5 * t); });
    Laplace1D L(G);
    Heat1DExplicit solver(0.12, L, bc);

    Field1D U = Field1D::from_fn(G, [](double x) {
        double dx = x - 0.5;
        return 295.0 + std::exp(-120.0 * dx * dx);
    });
    if (apply_geospatial(U, geodata)) {
        std::cout << "Applied geospatial data as the initial 1D temperature field.\n";
    }
    clean_field(U, weather.ambient_temp - 40.0, weather.ambient_temp + 40.0);

    double dt = 0.4 * G.dx * G.dx / 0.12;
    double total_time = 0.05;
    int steps = static_cast<int>(total_time / dt);
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        clean_field(U, weather.ambient_temp - 100.0, weather.ambient_temp + 100.0);
        t += dt;
    }

    auto ghosts = capture_ghost_cells(U);
    auto table = make_table(U, weather, t);
    return {std::move(U), ghosts, std::move(table), t};
}

SimulationResult<Field1D, GhostCells1D> run_heat_convection_1d(
    const WeatherModel& weather, const std::vector<double>& geodata) {
    Grid1D G(1.0, 256);
    DirichletBC1D bc(
        [&weather](double t) { return weather.ambient_temp; },
        [&weather](double t) { return weather.ambient_temp + 1.5 * std::cos(0.5 * t); });
    Laplace1D L(G);
    Advection1D adv([&weather](double x, double t) { return weather.wind_field(x, 0.0, 0.0, t)[0]; });
    AdvectionDiffusion1DExplicit solver(0.1, adv, L, bc);

    Field1D U = Field1D::from_fn(G, [](double x) {
        double dx = x - 0.3;
        return 290.0 + std::exp(-80.0 * dx * dx);
    });
    apply_geospatial(U, geodata);
    clean_field(U, weather.ambient_temp - 60.0, weather.ambient_temp + 60.0);

    double dt = 0.2 * G.dx;
    double total_time = 0.04;
    int steps = static_cast<int>(total_time / dt);
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        clean_field(U, weather.ambient_temp - 120.0, weather.ambient_temp + 120.0);
        t += dt;
    }
    auto ghosts = capture_ghost_cells(U);
    auto table = make_table(U, weather, t);
    return {std::move(U), ghosts, std::move(table), t};
}

SimulationResult<Field2D, GhostCells2D> run_heat_conduction_2d(
    const WeatherModel& weather, const std::vector<double>& geodata) {
    Grid2D G(1.0, 1.0, 160, 160);
    DirichletBC2D bc([&weather](double x, double y) {
        return weather.ambient_temp - 3.0 * weather.rain_rate(x, y, 0.0, 0.0);
    });
    Laplace2D L(G);
    Heat2D_ADI solver(G, bc, 0.2);

    Field2D U(G);
    auto fallback = [](double x, double y) {
        double cx = 0.5, cy = 0.5;
        double dx = x - cx, dy = y - cy;
        return 300.0 * std::exp(-80.0 * (dx * dx + dy * dy));
    };
    auto init = build_initializer(G, geodata, fallback);
    solver.solve(0.02, 5e-5, init, U);
    clean_field(U, weather.ambient_temp - 80.0, weather.ambient_temp + 80.0);

    auto ghosts = capture_ghost_cells(U);
    auto table = make_table(U, weather, 0.02);
    return {std::move(U), std::move(ghosts), std::move(table), 0.02};
}

SimulationResult<Field2D, GhostCells2D> run_heat_convection_2d(
    const WeatherModel& weather, const std::vector<double>& geodata) {
    Grid2D G(1.0, 1.0, 200, 200);
    DirichletBC2D bc([&weather](double x, double y) {
        return weather.ambient_temp - 1.5 * weather.rain_rate(x, y, 0.0, 0.0);
    });
    Laplace2D L(G);
    Advection2D A(
        [&weather](double x, double y, double t) { return weather.wind_field(x, y, 0.0, t)[0]; },
        [&weather](double x, double y, double t) { return weather.wind_field(x, y, 0.0, t)[1]; });
    AdvectionDiffusion2DExplicit solver(0.1, A, L, bc);
    Field2D U(G);
    for (int j = 0; j <= G.Ny; ++j)
        for (int i = 0; i <= G.Nx; ++i) {
            double dx = G.x[i] - 0.25;
            double dy = G.y[j] - 0.5;
            U.u[U.id(i, j)] = 285.0 + 15.0 * std::exp(-90.0 * (dx * dx + dy * dy));
        }
    apply_geospatial(U, geodata);
    clean_field(U, weather.ambient_temp - 70.0, weather.ambient_temp + 70.0);

    auto safe_cfl = [](double vel, double h) { return (std::abs(vel) < 1e-12) ? 1e9 : h / std::abs(vel); };
    double adv_dt = 0.4 * std::min(safe_cfl(0.6, G.dx), safe_cfl(0.25, G.dy));
    double diff_dt = 0.2 * std::min(G.dx * G.dx, G.dy * G.dy) / solver.a;
    double dt = std::min(adv_dt, diff_dt);
    int steps = 450;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        clean_field(U, weather.ambient_temp - 120.0, weather.ambient_temp + 120.0);
        t += dt;
    }
    auto ghosts = capture_ghost_cells(U);
    auto table = make_table(U, weather, t);
    return {std::move(U), std::move(ghosts), std::move(table), t};
}

SimulationResult<Field3D, GhostCells3D> run_heat_conduction_3d(
    const WeatherModel& weather, const std::vector<double>& geodata) {
    Grid3D G(1.0, 1.0, 1.0, 64, 64, 64);
    DirichletBC3D bc([&weather](double x, double y, double z) {
        return weather.ambient_temp - 2.0 * weather.rain_rate(x, y, z, 0.0);
    });
    Laplace3D L(G);
    Heat3DExplicit solver(0.08, L, bc);
    Field3D U(G);
    for (int k = 0; k <= G.Nz; ++k)
        for (int j = 0; j <= G.Ny; ++j)
            for (int i = 0; i <= G.Nx; ++i)
                U.u[U.id(i, j, k)] = 290.0
                    + 10.0 * std::sin(M_PI * G.x[i]) * std::sin(M_PI * G.y[j]) * std::sin(M_PI * G.z[k]);
    apply_geospatial(U, geodata);
    clean_field(U, weather.ambient_temp - 50.0, weather.ambient_temp + 50.0);

    double diff_dt = 0.15 * std::min({G.dx * G.dx, G.dy * G.dy, G.dz * G.dz}) / solver.a;
    double total_time = 0.01;
    int steps = static_cast<int>(total_time / diff_dt);
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, diff_dt);
        clean_field(U, weather.ambient_temp - 120.0, weather.ambient_temp + 120.0);
        t += diff_dt;
    }
    auto ghosts = capture_ghost_cells(U);
    auto table = make_table(U, weather, t);
    return {std::move(U), std::move(ghosts), std::move(table), t};
}

SimulationResult<Field3D, GhostCells3D> run_heat_convection_3d(
    const WeatherModel& weather, const std::vector<double>& geodata) {
    Grid3D G(1.0, 1.0, 1.0, 64, 64, 64);
    DirichletBC3D bc([&weather](double x, double y, double z) {
        return weather.ambient_temp - 1.0 * weather.rain_rate(x, y, z, 0.0);
    });
    Laplace3D L(G);
    Advection3D A(
        [&weather](double x, double y, double z, double t) { return weather.wind_field(x, y, z, t)[0]; },
        [&weather](double x, double y, double z, double t) { return weather.wind_field(x, y, z, t)[1]; },
        [&weather](double x, double y, double z, double t) { return weather.wind_field(x, y, z, t)[2]; });
    AdvectionDiffusion3DExplicit solver(0.06, A, L, bc);

    Field3D U(G);
    for (int k = 0; k <= G.Nz; ++k)
        for (int j = 0; j <= G.Ny; ++j)
            for (int i = 0; i <= G.Nx; ++i) {
                double bubble = std::exp(-90.0 * ((G.x[i] - 0.4) * (G.x[i] - 0.4)
                    + (G.y[j] - 0.4) * (G.y[j] - 0.4) + (G.z[k] - 0.5) * (G.z[k] - 0.5)));
                U.u[U.id(i, j, k)] = weather.ambient_temp + 15.0 * bubble;
            }
    apply_geospatial(U, geodata);
    clean_field(U, weather.ambient_temp - 80.0, weather.ambient_temp + 80.0);

    auto safe_cfl = [](double vel, double h) { return (std::abs(vel) < 1e-12) ? 1e9 : h / std::abs(vel); };
    double adv_dt = 0.3 * std::min({safe_cfl(0.5, G.dx), safe_cfl(0.3, G.dy), safe_cfl(0.2, G.dz)});
    double diff_dt = 0.15 * std::min({G.dx * G.dx, G.dy * G.dy, G.dz * G.dz}) / solver.a;
    double dt = std::min(adv_dt, diff_dt);
    int steps = 200;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        clean_field(U, weather.ambient_temp - 150.0, weather.ambient_temp + 150.0);
        t += dt;
    }
    auto ghosts = capture_ghost_cells(U);
    auto table = make_table(U, weather, t);
    return {std::move(U), std::move(ghosts), std::move(table), t};
}

template <class FieldType, class GhostType>
void write_artifacts(const std::string& mode, const SimulationResult<FieldType, GhostType>& res,
                     const CLIOptions& opts);

template <>
void write_artifacts(const std::string& mode, const SimulationResult<Field1D, GhostCells1D>& res,
                     const CLIOptions& opts) {
    std::string table_path = RESULTS_DIR + "/" + mode + "_table.csv";
    std::string ghost_path = RESULTS_DIR + "/" + mode + "_ghosts.csv";
    std::string vtk_path = opts.vtk_prefix.empty()
        ? (RESULTS_DIR + "/" + mode + ".vtk")
        : (opts.vtk_prefix + "_" + mode + ".vtk");
    write_table_csv(table_path, res.table);
    write_ghost_csv(ghost_path, res.ghosts);
    write_vtk(vtk_path, res.field);
}

template <>
void write_artifacts(const std::string& mode, const SimulationResult<Field2D, GhostCells2D>& res,
                     const CLIOptions& opts) {
    std::string table_path = RESULTS_DIR + "/" + mode + "_table.csv";
    std::string ghost_path = RESULTS_DIR + "/" + mode + "_ghosts.csv";
    std::string vtk_path = opts.vtk_prefix.empty()
        ? (RESULTS_DIR + "/" + mode + ".vtk")
        : (opts.vtk_prefix + "_" + mode + ".vtk");
    write_table_csv(table_path, res.table);
    write_ghost_csv(ghost_path, res.ghosts, res.field);
    write_vtk(vtk_path, res.field);
}

template <>
void write_artifacts(const std::string& mode, const SimulationResult<Field3D, GhostCells3D>& res,
                     const CLIOptions& opts) {
    std::string table_path = RESULTS_DIR + "/" + mode + "_table.csv";
    std::string ghost_path = RESULTS_DIR + "/" + mode + "_ghosts.csv";
    std::string vtk_path = opts.vtk_prefix.empty()
        ? (RESULTS_DIR + "/" + mode + ".vtk")
        : (opts.vtk_prefix + "_" + mode + ".vtk");
    write_table_csv(table_path, res.table);
    write_ghost_csv(ghost_path, res.ghosts, res.field);
    write_vtk(vtk_path, res.field);
}

void report_summary(const std::string& mode, const Field1D& U) {
    auto [mn_it, mx_it] = std::minmax_element(U.u.begin(), U.u.end());
    std::cout << mode << " -> min: " << *mn_it << ", max: " << *mx_it
              << ", avg: " << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << "\n";
}

void report_summary(const std::string& mode, const Field2D& U) {
    auto [mn_it, mx_it] = std::minmax_element(U.u.begin(), U.u.end());
    std::cout << mode << " -> min: " << *mn_it << ", max: " << *mx_it
              << ", avg: " << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << "\n";
}

void report_summary(const std::string& mode, const Field3D& U) {
    auto [mn_it, mx_it] = std::minmax_element(U.u.begin(), U.u.end());
    std::cout << mode << " -> min: " << *mn_it << ", max: " << *mx_it
              << ", avg: " << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << "\n";
}

int main(int argc, char** argv) {
    CLIOptions opts = parse_cli(argc, argv);
    fs::create_directories(RESULTS_DIR);

    WeatherModel weather = default_weather_model();
    auto geodata = load_geospatial_csv(opts.geodata_path);

    auto wall_start = std::chrono::high_resolution_clock::now();
    if (opts.mode == "conduction_1d") {
        auto res = run_heat_conduction_1d(weather, geodata);
        write_artifacts(opts.mode, res, opts);
        report_summary(opts.mode, res.field);
    } else if (opts.mode == "convection_1d") {
        auto res = run_heat_convection_1d(weather, geodata);
        write_artifacts(opts.mode, res, opts);
        report_summary(opts.mode, res.field);
    } else if (opts.mode == "conduction_2d") {
        auto res = run_heat_conduction_2d(weather, geodata);
        write_artifacts(opts.mode, res, opts);
        report_summary(opts.mode, res.field);
    } else if (opts.mode == "convection_2d") {
        auto res = run_heat_convection_2d(weather, geodata);
        write_artifacts(opts.mode, res, opts);
        report_summary(opts.mode, res.field);
    } else if (opts.mode == "conduction_3d") {
        auto res = run_heat_conduction_3d(weather, geodata);
        write_artifacts(opts.mode, res, opts);
        report_summary(opts.mode, res.field);
    } else if (opts.mode == "convection_3d") {
        auto res = run_heat_convection_3d(weather, geodata);
        write_artifacts(opts.mode, res, opts);
        report_summary(opts.mode, res.field);
    } else {
        std::cout << "Unknown mode: " << opts.mode << "\n";
        return 1;
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed wall time: " << std::chrono::duration<double>(wall_end - wall_start).count() << " s\n";
    return 0;
}
