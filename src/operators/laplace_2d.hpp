// 2D Laplacian operator.
#pragma once
#include <vector>
#include <omp.h>
#include "../grid_2d.hpp"

struct Laplace2D {
    const Grid2D* g;
    double dx2, dy2;
    explicit Laplace2D(const Grid2D& G) : g(&G), dx2(G.dx * G.dx), dy2(G.dy * G.dy) {}
    std::vector<double> apply(const Field2D& F) const {
        const auto& G = *g;
        std::vector<double> d2(F.u.size(), 0.0);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < G.Ny; ++j)
            for (int i = 1; i < G.Nx; ++i) {
                int id = F.id(i, j);
                const double uxx = (F.u[F.id(i + 1, j)] - 2.0 * F.u[id] + F.u[F.id(i - 1, j)]) / dx2;
                const double uyy = (F.u[F.id(i, j + 1)] - 2.0 * F.u[id] + F.u[F.id(i, j - 1)]) / dy2;
                d2[id] = uxx + uyy;
            }
        return d2;
    }
};
