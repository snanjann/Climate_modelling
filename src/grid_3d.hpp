// 3D uniform grid and scalar field.
#pragma once
#include <vector>
#include <functional>
#include "common.hpp"

struct Grid3D {
    double Lx, Ly, Lz;
    int Nx, Ny, Nz;
    double dx, dy, dz;
    std::vector<double> x, y, z;

    Grid3D(double Lx_, double Ly_, double Lz_, int Nx_, int Ny_, int Nz_)
        : Lx(Lx_), Ly(Ly_), Lz(Lz_), Nx(Nx_), Ny(Ny_), Nz(Nz_) {
        ensure(Nx >= 2 && Ny >= 2 && Nz >= 2, "Grid3D Nx,Ny,Nz>=2");
        dx = Lx / Nx;
        dy = Ly / Ny;
        dz = Lz / Nz;
        x.resize(Nx + 1);
        y.resize(Ny + 1);
        z.resize(Nz + 1);
        for (int i = 0; i <= Nx; ++i) x[i] = i * dx;
        for (int j = 0; j <= Ny; ++j) y[j] = j * dy;
        for (int k = 0; k <= Nz; ++k) z[k] = k * dz;
    }
};

struct Field3D {
    const Grid3D* g;
    std::vector<double> u;  // (Nz+1)*(Ny+1)*(Nx+1)
    Field3D(const Grid3D& G) : g(&G), u((size_t)(G.Nx + 1) * (G.Ny + 1) * (G.Nz + 1), 0.0) {}
    inline int id(int i, int j, int k) const {
        return (k * (g->Ny + 1) + j) * (g->Nx + 1) + i;
    }
};

struct DirichletBC3D {
    std::function<double(double, double, double)> f;
    template <class Fun>
    DirichletBC3D(Fun F) : f(F) {}
    void apply(Field3D& U) const {
        auto& G = *U.g;
        for (int j = 0; j <= G.Ny; ++j)
            for (int k = 0; k <= G.Nz; ++k) {
                U.u[U.id(0, j, k)] = f(0.0, G.y[j], G.z[k]);
                U.u[U.id(G.Nx, j, k)] = f(G.Lx, G.y[j], G.z[k]);
            }
        for (int i = 0; i <= G.Nx; ++i)
            for (int k = 0; k <= G.Nz; ++k) {
                U.u[U.id(i, 0, k)] = f(G.x[i], 0.0, G.z[k]);
                U.u[U.id(i, G.Ny, k)] = f(G.x[i], G.Ly, G.z[k]);
            }
        for (int i = 0; i <= G.Nx; ++i)
            for (int j = 0; j <= G.Ny; ++j) {
                U.u[U.id(i, j, 0)] = f(G.x[i], G.y[j], 0.0);
                U.u[U.id(i, j, G.Nz)] = f(G.x[i], G.y[j], G.Lz);
            }
    }
};
