// 1D Laplacian operator.
#pragma once
#include <vector>
#include <omp.h>
#include "../grid_1d.hpp"

struct Laplace1D {
    const Grid1D* g;
    double dx2;
    explicit Laplace1D(const Grid1D& G) : g(&G), dx2(G.dx * G.dx) {}
    std::vector<double> apply(const Field1D& F) const {
        int N = g->N;
        std::vector<double> d2(N + 1, 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < N; ++i)
            d2[i] = (F.u[i + 1] - 2.0 * F.u[i] + F.u[i - 1]) / dx2;
        return d2;
    }
};
