// 2D Heat equation solved with ADI (Crank-Nicolson split).
#pragma once
#include <vector>
#include <cmath>
#include <omp.h>
#include "../grid_2d.hpp"
#include "../operators/laplace_2d.hpp"

inline std::vector<double> thomas(std::vector<double> a,
                                  std::vector<double> b,
                                  std::vector<double> c,
                                  std::vector<double> d) {
    int n = static_cast<int>(d.size());
    for (int i = 1; i < n; ++i) {
        double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    std::vector<double> x(n, 0.0);
    x[n - 1] = d[n - 1] / b[n - 1];
    for (int i = n - 2; i >= 0; --i) x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
    return x;
}

struct Heat2D_ADI {
    const Grid2D* G;
    const DirichletBC2D* bc;
    double a;
    Heat2D_ADI(const Grid2D& g, const DirichletBC2D& b, double a_) : G(&g), bc(&b), a(a_) {}

    void solve(double T, double dt, std::function<double(double, double)> u0, Field2D& U) const {
        int Nx = G->Nx, Ny = G->Ny;
        for (int j = 0; j <= Ny; ++j)
            for (int i = 0; i <= Nx; ++i)
                U.u[U.id(i, j)] = u0(G->x[i], G->y[j]);
        bc->apply(U);

        double rx = a * dt / (G->dx * G->dx);
        double ry = a * dt / (G->dy * G->dy);
        int steps = static_cast<int>(std::round(T / dt));

        for (int n = 0; n < steps; ++n) {
            #pragma omp parallel for schedule(static)
            for (int j = 1; j < Ny; ++j) {
                int m = Nx - 1;
                std::vector<double> a_(m, -0.5 * rx), b(m, 1.0 + rx), c(m, -0.5 * rx), d(m, 0.0);
                for (int i = 1; i < Nx; ++i) {
                    double rhs = (1.0 - ry) * U.u[U.id(i, j)]
                        + 0.5 * ry * (U.u[U.id(i, j + 1)] + U.u[U.id(i, j - 1)]);
                    if (i == 1) rhs += 0.5 * rx * U.u[U.id(0, j)];
                    if (i == Nx - 1) rhs += 0.5 * rx * U.u[U.id(Nx, j)];
                    d[i - 1] = rhs;
                }
                auto s = thomas(a_, b, c, d);
                for (int i = 1; i < Nx; ++i) U.u[U.id(i, j)] = s[i - 1];
            }
            bc->apply(U);

            #pragma omp parallel for schedule(static)
            for (int i = 1; i < Nx; ++i) {
                int m = Ny - 1;
                std::vector<double> a_(m, -0.5 * ry), b(m, 1.0 + ry), c(m, -0.5 * ry), d(m, 0.0);
                for (int j = 1; j < Ny; ++j) {
                    double rhs = (1.0 - rx) * U.u[U.id(i, j)]
                        + 0.5 * rx * (U.u[U.id(i + 1, j)] + U.u[U.id(i - 1, j)]);
                    if (j == 1) rhs += 0.5 * ry * U.u[U.id(i, 0)];
                    if (j == Ny - 1) rhs += 0.5 * ry * U.u[U.id(i, Ny)];
                    d[j - 1] = rhs;
                }
                auto s = thomas(a_, b, c, d);
                for (int j = 1; j < Ny; ++j) U.u[U.id(i, j)] = s[j - 1];
            }
            bc->apply(U);
        }
    }
};
