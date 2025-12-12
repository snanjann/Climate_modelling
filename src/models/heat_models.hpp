#pragma once
#include <array>
#include <cmath>
#include <vector>

#include "common.hpp"
#include "grid_1d.hpp"
#include "grid_2d.hpp"
#include "grid_3d.hpp"
#include "operators/laplace_1d.hpp"
#include "operators/laplace_2d.hpp"
#include "operators/laplace_3d.hpp"
#include "operators/advection.hpp"
#include "solvers/heat_explicit.hpp"
#include "solvers/heat_adi.hpp"
#include "solvers/heat_3d_explicit.hpp"
#include "solvers/advection_diffusion.hpp"
#include "solvers/heat_with_forcing.hpp"
#include "utils/data.hpp"
#include "utils/geodata.hpp"
#include "utils/ghosts.hpp"
#include "utils/physics.hpp"
#include "utils/weather.hpp"

inline std::vector<GridPointRecord> make_table(const Field1D& U, const WeatherModel& weather, double time) {
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

inline std::vector<GridPointRecord> make_table(const Field2D& U, const WeatherModel& weather, double time) {
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

inline std::vector<GridPointRecord> make_table(const Field3D& U, const WeatherModel& weather, double time) {
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

inline SimulationResult<Field1D, GhostCells1D> run_heat_conduction_1d(
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

inline SimulationResult<Field1D, GhostCells1D> run_heat_convection_1d(
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

inline SimulationResult<Field2D, GhostCells2D> run_heat_conduction_2d(
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

inline SimulationResult<Field2D, GhostCells2D> run_heat_convection_2d(
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

inline SimulationResult<Field3D, GhostCells3D> run_heat_conduction_3d(
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

inline SimulationResult<Field3D, GhostCells3D> run_heat_convection_3d(
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

// ------------------------------------------------------------------
// Legacy/simple modes used for visualization: diffusion_2d,
// advection_2d, forcing_2d, full_climate_3d (explicit + forcing).
// These do not use weather/geodata and return just the field.
// ------------------------------------------------------------------

inline Field2D run_diffusion_2d_simple() {
    Grid2D G(1.0, 1.0, 128, 128);
    DirichletBC2D bc([](double, double) { return 0.0; });
    Laplace2D L(G);
    Heat2D_ADI solver(G, bc, 0.5);
    Field2D U(G);
    solver.solve(0.02, 5e-5, [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); }, U);
    return U;
}

inline Field2D run_advection_2d_simple() {
    Grid2D G(1.0, 1.0, 256, 256);
    DirichletBC2D bc([](double, double) { return 0.0; });
    Laplace2D L(G);
    Advection2D A([](double, double, double) { return 0.5; }, [](double, double, double) { return 0.0; });
    AdvectionDiffusion2DExplicit solver(0.1, A, L, bc);
    Field2D U(G);
    for (int j = 0; j <= G.Ny; ++j)
        for (int i = 0; i <= G.Nx; ++i) {
            double dx = G.x[i] - 0.25;
            double dy = G.y[j] - 0.5;
            U.u[U.id(i, j)] = std::exp(-80.0 * (dx * dx + dy * dy));
        }
    bc.apply(U);
    auto safe_cfl = [](double vel, double h) { return (std::abs(vel) < 1e-12) ? 1e9 : h / std::abs(vel); };
    double adv_dt = 0.5 * std::min(safe_cfl(0.5, G.dx), safe_cfl(0.0, G.dy));
    double diff_dt = 0.24 * std::min(G.dx * G.dx, G.dy * G.dy) / solver.a;
    double dt = std::min(adv_dt, diff_dt);
    int steps = 400;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    return U;
}

inline Field2D run_forcing_2d_simple() {
    Grid2D G(1.0, 1.0, 200, 200);
    DirichletBC2D bc([](double, double) { return 0.0; });
    Laplace2D L(G);
    Forcing2D Q([](double x, double y, double t) {
        double cx = 0.5 + 0.1 * std::sin(2 * M_PI * t);
        double cy = 0.5;
        double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
        return std::exp(-80.0 * r2);
    });
    Heat2DExplicitForcing solver(0.25, L, Q, bc);
    Field2D U(G);
    for (auto& v : U.u) v = 0.0;
    bc.apply(U);
    double dt = 0.24 * std::min(G.dx * G.dx, G.dy * G.dy) / solver.a;
    int steps = 300;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    return U;
}

inline Field3D run_full_climate_3d_simple() {
    Grid3D G(1.0, 1.0, 1.0, 64, 64, 64);
    DirichletBC3D bc([](double, double, double) { return 0.0; });
    Laplace3D L(G);
    Advection3D A([](double, double, double, double) { return 0.2; },
                  [](double, double, double, double) { return 0.0; },
                  [](double, double, double, double) { return 0.1; });
    Forcing3D Q([](double x, double y, double z, double t) {
        double cx = 0.5 + 0.05 * std::cos(2 * M_PI * t);
        double cy = 0.5;
        double cz = 0.5 + 0.05 * std::sin(2 * M_PI * t);
        double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
        return 0.5 * std::exp(-120.0 * r2);
    });
    Field3D U(G);
    for (int k = 0; k <= G.Nz; ++k)
        for (int j = 0; j <= G.Ny; ++j)
            for (int i = 0; i <= G.Nx; ++i)
                U.u[U.id(i, j, k)] = std::sin(M_PI * G.x[i]) * std::sin(M_PI * G.y[j]) * std::sin(M_PI * G.z[k]);
    bc.apply(U);
    auto safe_cfl3 = [](double vel, double h) { return (std::abs(vel) < 1e-12) ? 1e9 : h / std::abs(vel); };
    double adv_dt = 0.4 * std::min({safe_cfl3(0.2, G.dx), safe_cfl3(0.0, G.dy), safe_cfl3(0.1, G.dz)});
    double diff_dt = 0.16 * std::min({G.dx * G.dx, G.dy * G.dy, G.dz * G.dz}) / 0.1;
    double dt = std::min(adv_dt, diff_dt);
    int steps = 120;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        // Manual step: advection + diffusion + forcing
        Field3D V(*U.g);
        V.u = U.u;
        bc.apply(V);
        auto adv = A.apply(V, t);
        auto d2 = L.apply(V);
        auto q = Q.apply(V, t);
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < G.Nz; ++k)
            for (int j = 1; j < G.Ny; ++j)
                for (int i = 1; i < G.Nx; ++i) {
                    int id = V.id(i, j, k);
                    V.u[id] = V.u[id] + dt * (adv[id] + 0.1 * d2[id] + q[id]);
                }
        bc.apply(V);
        U = std::move(V);
        t += dt;
    }
    return U;
}
