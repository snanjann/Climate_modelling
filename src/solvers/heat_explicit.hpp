// 1D explicit heat equation solver.
#pragma once
#include <vector>
#include "../grid_1d.hpp"
#include "../operators/laplace_1d.hpp"

struct Heat1DExplicit {
    double a;
    const Laplace1D* L;
    const DirichletBC1D* bc;
    Heat1DExplicit(double a_, const Laplace1D& lap, const DirichletBC1D& b)
        : a(a_), L(&lap), bc(&b) {}

    Field1D step(const Field1D& u, double t, double dt) const {
        Field1D v = u;
        bc->apply(v, t);
        double r = a * dt / L->dx2;
        ensure(r <= 0.5, "Explicit unstable (1D)");
        auto d2 = L->apply(v);
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < u.g->N; ++i)
            v.u[i] = v.u[i] + dt * (a * d2[i]);
        bc->apply(v, t + dt);
        return v;
    }
};
