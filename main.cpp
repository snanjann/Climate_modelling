#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <numeric>
#include <algorithm>
#include <vector>
#include <omp.h>

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

void run_diffusion_2d() {
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
}

void run_advection_2d() {
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
    double diff_dt = 0.4 * std::min(G.dx * G.dx, G.dy * G.dy) / solver.a;     // diffusion
    double dt = std::min(adv_dt, diff_dt);
    int steps = 400;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    std::cout << "Peak value after advection: " << *std::max_element(U.u.begin(), U.u.end()) << std::endl;
}

void run_forcing_2d() {
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

    double dt = 0.4 * std::min(G.dx * G.dx, G.dy * G.dy) / solver.a;
    int steps = 300;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    std::cout << "Average temperature after forcing: "
              << std::accumulate(U.u.begin(), U.u.end(), 0.0) / U.u.size() << std::endl;
}

void run_full_climate_3d() {
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
    double diff_dt = 0.2 * std::min({G.dx * G.dx, G.dy * G.dy, G.dz * G.dz}) / solver.a;
    double dt = std::min(adv_dt, diff_dt);
    int steps = 120;
    double t = 0.0;
    for (int n = 0; n < steps; ++n) {
        U = solver.step(U, t, dt);
        t += dt;
    }
    int jmid = G.Ny / 2, kmid = G.Nz / 2;
    std::cout << "Centerline value (x in [0,1], y=0.5,z=0.5): " << U.u[U.id(G.Nx / 2, jmid, kmid)] << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 3 || std::string(argv[1]) != "--mode") {
        std::cout << "Usage: ./solver --mode [diffusion_2d|advection_2d|forcing_2d|full_climate_3d]\n";
        return 1;
    }
    std::string mode = argv[2];
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
