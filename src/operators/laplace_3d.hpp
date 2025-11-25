// 3D Laplacian operator (matrix-free).
#pragma once
#include <vector>
#include <omp.h>
#include "../grid_3d.hpp"

struct Laplace3D {
    const Grid3D* g;
    double dx2, dy2, dz2;
    explicit Laplace3D(const Grid3D& G) : g(&G), dx2(G.dx * G.dx), dy2(G.dy * G.dy), dz2(G.dz * G.dz) {}
    std::vector<double> apply(const Field3D& F) const {
        const auto& G = *g;
        std::vector<double> d2(F.u.size(), 0.0);
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < G.Nz; ++k)
            for (int j = 1; j < G.Ny; ++j)
                for (int i = 1; i < G.Nx; ++i) {
                    const int id = F.id(i, j, k);
                    const double uxx = (F.u[F.id(i + 1, j, k)] - 2.0 * F.u[id] + F.u[F.id(i - 1, j, k)]) / dx2;
                    const double uyy = (F.u[F.id(i, j + 1, k)] - 2.0 * F.u[id] + F.u[F.id(i, j - 1, k)]) / dy2;
                    const double uzz = (F.u[F.id(i, j, k + 1)] - 2.0 * F.u[id] + F.u[F.id(i, j, k - 1)]) / dz2;
                    d2[id] = uxx + uyy + uzz;
                }
        return d2;
    }
};
