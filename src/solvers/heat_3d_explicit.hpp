// 3D explicit heat equation (diffusion only).
#pragma once
#include "../grid_3d.hpp"
#include "../operators/laplace_3d.hpp"

struct Heat3DExplicit {
    double a;
    const Laplace3D* L;
    const DirichletBC3D* bc;
    Heat3DExplicit(double a_, const Laplace3D& lap, const DirichletBC3D& b)
        : a(a_), L(&lap), bc(&b) {}

    Field3D step(const Field3D& U, double t, double dt) const {
        Field3D V(*U.g);
        V.u = U.u;
        bc->apply(V);
        const auto& G = *U.g;
        double rsum = a * dt * (1.0 / (L->dx2) + 1.0 / (L->dy2) + 1.0 / (L->dz2));
        ensure(rsum <= 0.5 + 1e-12, "3D explicit unstable: reduce dt");
        auto d2 = L->apply(V);
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < G.Nz; ++k)
            for (int j = 1; j < G.Ny; ++j)
                for (int i = 1; i < G.Nx; ++i) {
                    int id = V.id(i, j, k);
                    V.u[id] = V.u[id] + dt * (a * d2[id]);
                }
        bc->apply(V);
        return V;
    }
};
