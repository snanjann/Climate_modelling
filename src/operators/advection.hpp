// Advection operators (first-order upwind) for 1D/2D/3D.
#pragma once
#include <vector>
#include <functional>
#include <omp.h>
#include "../grid_1d.hpp"
#include "../grid_2d.hpp"
#include "../grid_3d.hpp"

inline double upwind_derivative(double vel, double forward, double center, double backward, double h) {
    return (vel >= 0.0) ? (center - backward) / h : (forward - center) / h;
}

struct Advection1D {
    std::function<double(double, double)> u_vel; // velocity as f(x,t)
    template <class Fun>
    explicit Advection1D(Fun u) : u_vel(std::move(u)) {}

    std::vector<double> apply(const Field1D& F, double t) const {
        const auto& G = *F.g;
        std::vector<double> adv(F.u.size(), 0.0);
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < G.N; ++i) {
            double vel = u_vel(G.x[i], t);
            adv[i] = -vel * upwind_derivative(vel, F.u[i + 1], F.u[i], F.u[i - 1], G.dx);
        }
        return adv;
    }
};

struct Advection2D {
    std::function<double(double, double, double)> u_vel; // x-velocity
    std::function<double(double, double, double)> v_vel; // y-velocity
    Advection2D(std::function<double(double, double, double)> u,
                std::function<double(double, double, double)> v)
        : u_vel(std::move(u)), v_vel(std::move(v)) {}

    std::vector<double> apply(const Field2D& F, double t) const {
        const auto& G = *F.g;
        std::vector<double> adv(F.u.size(), 0.0);
        #pragma omp parallel for collapse(2) schedule(static)
        for (int j = 1; j < G.Ny; ++j)
            for (int i = 1; i < G.Nx; ++i) {
                const int id = F.id(i, j);
                double u = u_vel(G.x[i], G.y[j], t);
                double v = v_vel(G.x[i], G.y[j], t);
                double dTx = upwind_derivative(u, F.u[F.id(i + 1, j)], F.u[id], F.u[F.id(i - 1, j)], G.dx);
                double dTy = upwind_derivative(v, F.u[F.id(i, j + 1)], F.u[id], F.u[F.id(i, j - 1)], G.dy);
                adv[id] = -(u * dTx + v * dTy);
            }
        return adv;
    }
};

struct Advection3D {
    std::function<double(double, double, double, double)> u_vel;
    std::function<double(double, double, double, double)> v_vel;
    std::function<double(double, double, double, double)> w_vel;
    Advection3D(std::function<double(double, double, double, double)> u,
                std::function<double(double, double, double, double)> v,
                std::function<double(double, double, double, double)> w)
        : u_vel(std::move(u)), v_vel(std::move(v)), w_vel(std::move(w)) {}

    std::vector<double> apply(const Field3D& F, double t) const {
        const auto& G = *F.g;
        std::vector<double> adv(F.u.size(), 0.0);
        #pragma omp parallel for collapse(3) schedule(static)
        for (int k = 1; k < G.Nz; ++k)
            for (int j = 1; j < G.Ny; ++j)
                for (int i = 1; i < G.Nx; ++i) {
                    const int id = F.id(i, j, k);
                    double u = u_vel(G.x[i], G.y[j], G.z[k], t);
                    double v = v_vel(G.x[i], G.y[j], G.z[k], t);
                    double w = w_vel(G.x[i], G.y[j], G.z[k], t);
                    double dTx = upwind_derivative(u, F.u[F.id(i + 1, j, k)], F.u[id], F.u[F.id(i - 1, j, k)], G.dx);
                    double dTy = upwind_derivative(v, F.u[F.id(i, j + 1, k)], F.u[id], F.u[F.id(i, j - 1, k)], G.dy);
                    double dTz = upwind_derivative(w, F.u[F.id(i, j, k + 1)], F.u[id], F.u[F.id(i, j, k - 1)], G.dz);
                    adv[id] = -(u * dTx + v * dTy + w * dTz);
                }
        return adv;
    }
};
