// 1D uniform grid and scalar field.
#pragma once
#include <vector>
#include <functional>
#include "common.hpp"

struct Grid1D {
    double L;
    int N;
    double dx;
    std::vector<double> x;

    Grid1D(double L_, int N_) : L(L_), N(N_) {
        ensure(N >= 2, "Grid1D N>=2");
        dx = L / N;
        x.resize(N + 1);
        for (int i = 0; i <= N; ++i) x[i] = i * dx;
    }
};

struct Field1D {
    const Grid1D* g;
    std::vector<double> u;
    Field1D(const Grid1D& grid) : g(&grid), u(grid.N + 1, 0.0) {}

    template <class Fun>
    static Field1D from_fn(const Grid1D& g, Fun f) {
        Field1D F(g);
        for (int i = 0; i <= g.N; ++i) F.u[i] = f(g.x[i]);
        return F;
    }
};

struct DirichletBC1D {
    std::function<double(double)> left, right;
    template <class L, class R>
    DirichletBC1D(L l, R r) : left(l), right(r) {}
    void apply(Field1D& u, double t) const {
        u.u.front() = left(t);
        u.u.back() = right(t);
    }
};
