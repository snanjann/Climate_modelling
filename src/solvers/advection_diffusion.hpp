// Advection-diffusion solvers (explicit) in 2D and 3D.
#pragma once
#include <algorithm>
#include <cmath>
#include <omp.h>
#include "../grid_2d.hpp"
#include "../grid_3d.hpp"
#include "../operators/advection.hpp"
#include "../operators/laplace_2d.hpp"
#include "../operators/laplace_3d.hpp"
#include "../operators/forcing.hpp"

struct AdvectionDiffusion2DExplicit {
    double a;
    const Advection2D* A;
    const Laplace2D* L;
    const DirichletBC2D* bc;
    AdvectionDiffusion2DExplicit(double a_, const Advection2D& adv, const Laplace2D& lap, const DirichletBC2D& b)
        : a(a_), A(&adv), L(&lap), bc(&b) {}

    Field2D step(const Field2D& U, double t, double dt) const {
        Field2D V(*U.g);
        V.u = U.u;
        bc->apply(V);
        const auto& G = *U.g;
        double diff_cfl = a * dt * (1.0 / (L->dx2) + 1.0 / (L->dy2));
        ensure(diff_cfl <= 0.5 + 1e-12, "Diffusion CFL violated in 2D");

        // Conservative check for advection CFL based on local velocities.
        double max_ratio = 0.0;
        for (int j = 1; j < G.Ny; ++j)
            for (int i = 1; i < G.Nx; ++i) {
                double u = A->u_vel(G.x[i], G.y[j], t);
                double v = A->v_vel(G.x[i], G.y[j], t);
                max_ratio = std::max(max_ratio, std::abs(u) / G.dx + std::abs(v) / G.dy);
            }
        ensure(dt * max_ratio <= 1.0 + 1e-12, "Advection CFL violated in 2D");

        auto adv = A->apply(V, t);
        auto d2 = L->apply(V);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < G.Ny; ++j)
            for (int i = 1; i < G.Nx; ++i) {
                int id = V.id(i, j);
                V.u[id] = V.u[id] + dt * (adv[id] + a * d2[id]);
            }
        bc->apply(V);
        return V;
    }
};

// Full climate: advection + diffusion + forcing in 3D.
struct Climate3DExplicit {
    double a;
    const Advection3D* A;
    const Laplace3D* L;
    const Forcing3D* Q;
    const DirichletBC3D* bc;
    Climate3DExplicit(double a_, const Advection3D& adv, const Laplace3D& lap,
                      const Forcing3D& q, const DirichletBC3D& b)
        : a(a_), A(&adv), L(&lap), Q(&q), bc(&b) {}

    Field3D step(const Field3D& U, double t, double dt) const {
        Field3D V(*U.g);
        V.u = U.u;
        bc->apply(V);
        const auto& G = *U.g;
        double diff_cfl = a * dt * (1.0 / (L->dx2) + 1.0 / (L->dy2) + 1.0 / (L->dz2));
        ensure(diff_cfl <= 0.5 + 1e-12, "Diffusion CFL violated in 3D");

        double max_ratio = 0.0;
        for (int k = 1; k < G.Nz; ++k)
            for (int j = 1; j < G.Ny; ++j)
                for (int i = 1; i < G.Nx; ++i) {
                    double u = A->u_vel(G.x[i], G.y[j], G.z[k], t);
                    double v = A->v_vel(G.x[i], G.y[j], G.z[k], t);
                    double w = A->w_vel(G.x[i], G.y[j], G.z[k], t);
                    double ratio = std::abs(u) / G.dx + std::abs(v) / G.dy + std::abs(w) / G.dz;
                    if (ratio > max_ratio) max_ratio = ratio;
                }
        ensure(dt * max_ratio <= 1.0 + 1e-12, "Advection CFL violated in 3D");

        auto adv = A->apply(V, t);
        auto d2 = L->apply(V);
        auto q = Q->apply(V, t);
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < G.Nz; ++k)
            for (int j = 1; j < G.Ny; ++j)
                for (int i = 1; i < G.Nx; ++i) {
                    int id = V.id(i, j, k);
                    V.u[id] = V.u[id] + dt * (adv[id] + a * d2[id] + q[id]);
                }
        bc->apply(V);
        return V;
    }
};
