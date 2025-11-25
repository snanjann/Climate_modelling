// Forcing term Q(x,y,z,t) or lower-dimensional variants.
#pragma once
#include <vector>
#include <functional>
#include "../grid_2d.hpp"
#include "../grid_3d.hpp"

struct Forcing2D {
    std::function<double(double, double, double)> Q;
    template <class Fun>
    Forcing2D(Fun f) : Q(f) {}
    std::vector<double> apply(const Field2D& F, double t) const {
        const auto& G = *F.g;
        std::vector<double> q(F.u.size(), 0.0);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 0; j <= G.Ny; ++j)
            for (int i = 0; i <= G.Nx; ++i)
                q[F.id(i, j)] = Q(G.x[i], G.y[j], t);
        return q;
    }
};

struct Forcing3D {
    std::function<double(double, double, double, double)> Q;
    template <class Fun>
    Forcing3D(Fun f) : Q(f) {}
    std::vector<double> apply(const Field3D& F, double t) const {
        const auto& G = *F.g;
        std::vector<double> q(F.u.size(), 0.0);
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k <= G.Nz; ++k)
            for (int j = 0; j <= G.Ny; ++j)
                for (int i = 0; i <= G.Nx; ++i)
                    q[F.id(i, j, k)] = Q(G.x[i], G.y[j], G.z[k], t);
        return q;
    }
};
