#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include <vector>
#include <omp.h>
#include <fstream>
#include <filesystem>

namespace fs = std::__fs::filesystem;

#include "src/common.hpp"
#include "src/grid_1d.hpp"
#include "src/grid_2d.hpp"
#include "src/grid_3d.hpp"

#include "src/operators/laplace_1d.hpp"
#include "src/operators/laplace_2d.hpp"
#include "src/operators/laplace_3d.hpp"
#include "src/operators/advection.hpp"
#include "src/operators/forcing.hpp"

#include "src/solvers/heat_explicit.hpp"
#include "src/solvers/heat_adi.hpp"
#include "src/solvers/heat_3d_explicit.hpp"
#include "src/solvers/advection_diffusion.hpp"
#include "src/solvers/heat_with_forcing.hpp"

struct RunMeta {
    std::string mode;
    int threads;
    int Nx, Ny, Nz;
    double dt;
    int steps;
    double runtime;
};

struct Stats2D {
    double min, max, mean, mass, l2;
    double peak_value;
    double peak_x, peak_y;
};

struct Stats3D {
    double min, max, mean, mass, l2;
    double peak_value;
    double peak_x, peak_y, peak_z;
};

Stats2D compute_stats(const Field2D& U) {
    const auto& G = *U.g;
    double mn = U.u[0], mx = U.u[0], mass = 0.0, l2 = 0.0;
    double peak_v = U.u[0], peak_x = G.x[0], peak_y = G.y[0];
    size_t count = 0;
    for (int j = 0; j <= G.Ny; ++j) {
        for (int i = 0; i <= G.Nx; ++i) {
            const double v = U.u[U.id(i, j)];
            mass += v;
            l2 += v * v;
            mn = std::min(mn, v);
            if (v > mx) {
                mx = v;
                peak_v = v;
                peak_x = G.x[i];
                peak_y = G.y[j];
            }
            ++count;
        }
    }
    double mean = mass / static_cast<double>(count);
    return {mn, mx, mean, mass, l2, peak_v, peak_x, peak_y};
}

Stats3D compute_stats(const Field3D& U) {
    const auto& G = *U.g;
    double mn = U.u[0], mx = U.u[0], mass = 0.0, l2 = 0.0;
    double peak_v = U.u[0], peak_x = G.x[0], peak_y = G.y[0], peak_z = G.z[0];
    size_t count = 0;
    for (int k = 0; k <= G.Nz; ++k) {
        for (int j = 0; j <= G.Ny; ++j) {
            for (int i = 0; i <= G.Nx; ++i) {
                const double v = U.u[U.id(i, j, k)];
                mass += v;
                l2 += v * v;
                mn = std::min(mn, v);
                if (v > mx) {
                    mx = v;
                    peak_v = v;
                    peak_x = G.x[i];
                    peak_y = G.y[j];
                    peak_z = G.z[k];
                }
                ++count;
            }
        }
    }
    double mean = mass / static_cast<double>(count);
    return {mn, mx, mean, mass, l2, peak_v, peak_x, peak_y, peak_z};
}

void ensure_log_header(const std::string& path, bool is3d) {
    if (fs::exists(path)) return;
    std::ofstream f(path);
    f << "mode,threads,Nx,Ny,Nz,dt,steps,runtime_s,min,mean,max,mass,l2,peak,peak_x,peak_y";
    if (is3d) f << ",peak_z";
    f << "\n";
}

void append_run_log(const std::string& path, const RunMeta& meta, const Stats2D& s) {
    ensure_log_header(path, false);
    std::ofstream f(path, std::ios::app);
    f << meta.mode << "," << meta.threads << "," << meta.Nx << "," << meta.Ny << "," << meta.Nz << ","
      << meta.dt << "," << meta.steps << "," << meta.runtime << ","
      << s.min << "," << s.mean << "," << s.max << "," << s.mass << "," << s.l2 << ","
      << s.peak_value << "," << s.peak_x << "," << s.peak_y << "\n";
}

void append_run_log(const std::string& path, const RunMeta& meta, const Stats3D& s) {
    ensure_log_header(path, true);
    std::ofstream f(path, std::ios::app);
    f << meta.mode << "," << meta.threads << "," << meta.Nx << "," << meta.Ny << "," << meta.Nz << ","
      << meta.dt << "," << meta.steps << "," << meta.runtime << ","
      << s.min << "," << s.mean << "," << s.max << "," << s.mass << "," << s.l2 << ","
      << s.peak_value << "," << s.peak_x << "," << s.peak_y << "," << s.peak_z << "\n";
}

void write_centerline_2d(const Field2D& U, const std::string& path) {
    const auto& G = *U.g;
    int jmid = G.Ny / 2;
    std::ofstream f(path);
    f << "x,u\n";
    for (int i = 0; i <= G.Nx; ++i) {
        f << G.x[i] << "," << U.u[U.id(i, jmid)] << "\n";
    }
}

void write_centerline_3d(const Field3D& U, const std::string& path) {
    const auto& G = *U.g;
    int jmid = G.Ny / 2, kmid = G.Nz / 2;
    std::ofstream f(path);
    f << "x,u\n";
    for (int i = 0; i <= G.Nx; ++i) {
        f << G.x[i] << "," << U.u[U.id(i, jmid, kmid)] << "\n";
    }
}

const std::string RESULTS_DIR = "Results_updated";

void run_diffusion_2d() {
    auto wall_start = std::chrono::high_resolution_clock::now();
    std::cout << "Running 2D diffusion (ADI)..." << std::endl;
    Grid2D G(1.0, 1.0, 128, 128);
    DirichletBC2D bc([](double, double) { return 0.0; });
    Laplace2D L(G);
    Heat2D_ADI solver(G, bc, 0.5);

    double T = 0.02, dt = 5e-5;
    Field2D U(G);
    solver.solve(T, dt, [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); }, U);
    int jmid = G.Ny / 2;
    std::cout << "Centerline value at x=0.5,y=0.5: " << U.u[U.id(G.Nx / 2, jmid)] << std::endl;

    auto stats = compute_stats(U);
    double runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_start).count();
    RunMeta meta{"diffusion_2d", omp_get_max_threads(), G.Nx, G.Ny, 0, dt, static_cast<int>(T / dt), runtime};
    append_run_log(RESULTS_DIR + "/run_log_2d.csv", meta, stats);
    write_centerline_2d(U, RESULTS_DIR + "/diffusion_2d_centerline_t" + std::to_string(meta.threads) + ".csv");
}

void run_advection_2d() {
    auto wall_start = std::chrono::high_resolution_clock::now();
    std::cout << "Running 2D advection-diffusion (explicit)..." << std::endl;
    Grid2D G(1.0, 1.0, 256, 256);
    DirichletBC2D bc([](double, double) { return 0.0; });
    Laplace2D L(G);
    Advection2D A(
        [](double, double, double) { return 0.5; },  // u
        [](double, double, double) { return 0.0; }   // v
    );
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
    double adv_dt = 0.5 * std::min(safe_cfl(0.5, G.dx), safe_cfl(0.0, G.dy)); // CFL for advection
    // Diffusion CFL needs a stricter factor: a*dt*(1/dx^2+1/dy^2) <= 0.5.
    double diff_dt = 0.24 * std::min(G.dx * G.dx, G.dy * G.dy) / solver.a;
    double dt = std::min(adv_dt, diff_dt);
    int steps = 400;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    std::cout << "Peak value after advection: " << *std::max_element(U.u.begin(), U.u.end()) << std::endl;

    auto stats = compute_stats(U);
    double runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_start).count();
    RunMeta meta{"advection_2d", omp_get_max_threads(), G.Nx, G.Ny, 0, dt, steps, runtime};
    append_run_log(RESULTS_DIR + "/run_log_2d.csv", meta, stats);
    write_centerline_2d(U, RESULTS_DIR + "/advection_2d_centerline_t" + std::to_string(meta.threads) + ".csv");
}

void run_forcing_2d() {
    auto wall_start = std::chrono::high_resolution_clock::now();
    std::cout << "Running 2D diffusion with forcing (explicit)..." << std::endl;
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

    // Ensure diffusion CFL: a*dt*(1/dx^2+1/dy^2) <= 0.5 with some margin.
    double dt = 0.24 * std::min(G.dx * G.dx, G.dy * G.dy) / solver.a;
    int steps = 300;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    std::cout << "Average temperature after forcing: "
              << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << std::endl;

    auto stats = compute_stats(U);
    double runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_start).count();
    RunMeta meta{"forcing_2d", omp_get_max_threads(), G.Nx, G.Ny, 0, dt, steps, runtime};
    append_run_log(RESULTS_DIR + "/run_log_2d.csv", meta, stats);
    write_centerline_2d(U, RESULTS_DIR + "/forcing_2d_centerline_t" + std::to_string(meta.threads) + ".csv");
}

void run_full_climate_3d() {
    auto wall_start = std::chrono::high_resolution_clock::now();
    std::cout << "Running 3D full climate (advection + diffusion + forcing)..." << std::endl;
    Grid3D G(1.0, 1.0, 1.0, 64, 64, 64);
    DirichletBC3D bc([](double, double, double) { return 0.0; });
    Laplace3D L(G);
    Advection3D A(
        [](double, double, double, double) { return 0.2; },
        [](double, double, double, double) { return 0.0; },
        [](double, double, double, double) { return 0.1; }
    );
    Forcing3D Q([](double x, double y, double z, double t) {
        double cx = 0.5 + 0.05 * std::cos(2 * M_PI * t);
        double cy = 0.5;
        double cz = 0.5 + 0.05 * std::sin(2 * M_PI * t);
        double r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy) + (z - cz) * (z - cz);
        return 0.5 * std::exp(-120.0 * r2);
    });
    Climate3DExplicit solver(0.1, A, L, Q, bc);

    Field3D U(G);
    for (int k = 0; k <= G.Nz; ++k)
        for (int j = 0; j <= G.Ny; ++j)
            for (int i = 0; i <= G.Nx; ++i)
                U.u[U.id(i, j, k)] = std::sin(M_PI * G.x[i]) * std::sin(M_PI * G.y[j]) * std::sin(M_PI * G.z[k]);
    bc.apply(U);

    auto safe_cfl3 = [](double vel, double h) { return (std::abs(vel) < 1e-12) ? 1e9 : h / std::abs(vel); };
    double adv_dt = 0.4 * std::min({safe_cfl3(0.2, G.dx), safe_cfl3(0.0, G.dy), safe_cfl3(0.1, G.dz)}); // advection CFL
    // For equal spacing, explicit 3D diffusion is stable if a*dt*(1/dx^2+1/dy^2+1/dz^2) <= 0.5 (~dt <= dx^2/(6a)).
    double diff_dt = 0.16 * std::min({G.dx * G.dx, G.dy * G.dy, G.dz * G.dz}) / solver.a;
    double dt = std::min(adv_dt, diff_dt);
    int steps = 120;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    int jmid = G.Ny / 2, kmid = G.Nz / 2;
    std::cout << "Centerline value (x in [0,1], y=0.5,z=0.5): " << U.u[U.id(G.Nx / 2, jmid, kmid)] << std::endl;

    auto stats = compute_stats(U);
    double runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_start).count();
    RunMeta meta{"full_climate_3d", omp_get_max_threads(), G.Nx, G.Ny, G.Nz, dt, steps, runtime};
    append_run_log(RESULTS_DIR + "/run_log_3d.csv", meta, stats);
    write_centerline_3d(U, RESULTS_DIR + "/full_climate_3d_centerline_t" + std::to_string(meta.threads) + ".csv");
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--mode") {
        std::cout << "Usage: ./solver --mode [diffusion_2d|advection_2d|forcing_2d|full_climate_3d]\n";
        return 1;
    }
    std::string mode = argv[2];
    fs::create_directories(RESULTS_DIR);
    auto start = std::chrono::high_resolution_clock::now();
    if (mode == "diffusion_2d") {
        run_diffusion_2d();
    } else if (mode == "advection_2d") {
        run_advection_2d();
    } else if (mode == "forcing_2d") {
        run_forcing_2d();
    } else if (mode == "full_climate_3d") {
        run_full_climate_3d();
    } else {
        std::cout << "Unknown mode: " << mode << std::endl;
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed: " << std::chrono::duration<double>(end - start).count() << " s" << std::endl;
    return 0;
}
