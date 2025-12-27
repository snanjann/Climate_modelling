#include "heat_models.hpp"

#include <cmath>
#include <omp.h>

#include "operators/advection.hpp"
#include "operators/forcing.hpp"
#include "operators/laplace_2d.hpp"
#include "operators/laplace_3d.hpp"
#include "solvers/advection_diffusion.hpp"
#include "solvers/heat_adi.hpp"
#include "solvers/heat_with_forcing.hpp"

Field2D run_diffusion_2d_simple() {
    Grid2D G(1.0, 1.0, 128, 128);
    DirichletBC2D bc([](double, double) { return 0.0; });
    Laplace2D L(G);
    Heat2D_ADI solver(G, bc, 0.5);
    Field2D U(G);
    solver.solve(0.02, 5e-5, [](double x, double y) { return std::sin(M_PI * x) * std::sin(M_PI * y); }, U);
    return U;
}

Field2D run_advection_2d_simple() {
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

Field2D run_forcing_2d_simple() {
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
        Field2D V(*U.g);
        V.u = U.u;
        bc.apply(V);
        double rsum = solver.a * dt * (1.0 / (L.dx2) + 1.0 / (L.dy2));
        ensure(rsum <= 0.5 + 1e-12, "2D explicit unstable: reduce dt");
        auto d2 = L.apply(V);
        auto q = Q.apply(V, t);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < G.Ny; ++j)
            for (int i = 1; i < G.Nx; ++i) {
                int id = V.id(i, j);
                V.u[id] = V.u[id] + dt * (solver.a * d2[id] + q[id]);
            }
        bc.apply(V);
        U = std::move(V);
        t += dt;
    }
    return U;
}

Field3D run_full_climate_3d_simple() {
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
