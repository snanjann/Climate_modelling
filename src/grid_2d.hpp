// 2D uniform grid and scalar field.
#pragma once
#include <vector>
#include <functional>
#include "common.hpp"

struct Grid2D {
    double Lx, Ly;
    int Nx, Ny;
    double dx, dy;
    std::vector<double> x, y;

    Grid2D(double Lx_, double Ly_, int Nx_, int Ny_)
        : Lx(Lx_), Ly(Ly_), Nx(Nx_), Ny(Ny_) {
        ensure(Nx >= 2 && Ny >= 2, "Grid2D Nx,Ny>=2");
        dx = Lx / Nx;
        dy = Ly / Ny;
        x.resize(Nx + 1);
        y.resize(Ny + 1);
        for (int i = 0; i <= Nx; ++i) x[i] = i * dx;
        for (int j = 0; j <= Ny; ++j) y[j] = j * dy;
    }
};

struct Field2D {
    const Grid2D* g;
    std::vector<double> u;
    Field2D(const Grid2D& G) : g(&G), u((G.Nx + 1) * (G.Ny + 1), 0.0) {}
    inline int id(int i, int j) const { return j * (g->Nx + 1) + i; }
};

struct DirichletBC2D {
    std::function<double(double, double)> f;
    template <class Fun>
    DirichletBC2D(Fun F) : f(F) {}
    void apply(Field2D& U) const {
        auto& G = *U.g;
        for (int i = 0; i <= G.Nx; ++i) {
            U.u[U.id(i, 0)] = f(G.x[i], 0.0);
            U.u[U.id(i, G.Ny)] = f(G.x[i], G.Ly);
        }
        for (int j = 0; j <= G.Ny; ++j) {
            U.u[U.id(0, j)] = f(0.0, G.y[j]);
            U.u[U.id(G.Nx, j)] = f(G.Lx, G.y[j]);
        }
    }
};
