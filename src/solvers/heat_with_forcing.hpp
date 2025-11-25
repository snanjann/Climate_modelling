// 2D explicit heat equation with forcing term Q.
#pragma once
#include "../grid_2d.hpp"
#include "../operators/laplace_2d.hpp"
#include "../operators/forcing.hpp"

struct Heat2DExplicitForcing {
    double a;
    const Laplace2D* L;
    const Forcing2D* Q;
    const DirichletBC2D* bc;
    Heat2DExplicitForcing(double a_, const Laplace2D& lap, const Forcing2D& q, const DirichletBC2D& b)
        : a(a_), L(&lap), Q(&q), bc(&b) {}

    Field2D step(const Field2D& U, double t, double dt) const {
        Field2D V(*U.g);
        V.u = U.u;
        bc->apply(V);
        const auto& G = *U.g;
        double rsum = a * dt * (1.0 / (L->dx2) + 1.0 / (L->dy2));
        ensure(rsum <= 0.5 + 1e-12, "2D explicit unstable: reduce dt");
        auto d2 = L->apply(V);
        auto q = Q->apply(V, t);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < G.Ny; ++j)
            for (int i = 1; i < G.Nx; ++i) {
                int id = V.id(i, j);
                V.u[id] = V.u[id] + dt * (a * d2[id] + q[id]);
            }
        bc->apply(V);
        return V;
    }
};
