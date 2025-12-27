#include "ghosts.hpp"

GhostCells1D capture_ghost_cells(const Field1D& U) {
    return {U.u.front(), U.u.back()};
}

GhostCells2D capture_ghost_cells(const Field2D& U) {
    const auto& G = *U.g;
    GhostCells2D ghosts;
    ghosts.west.resize(G.Ny + 1);
    ghosts.east.resize(G.Ny + 1);
    ghosts.south.resize(G.Nx + 1);
    ghosts.north.resize(G.Nx + 1);
    for (int j = 0; j <= G.Ny; ++j) {
        ghosts.west[j] = U.u[U.id(0, j)];
        ghosts.east[j] = U.u[U.id(G.Nx, j)];
    }
    for (int i = 0; i <= G.Nx; ++i) {
        ghosts.south[i] = U.u[U.id(i, 0)];
        ghosts.north[i] = U.u[U.id(i, G.Ny)];
    }
    return ghosts;
}

GhostCells3D capture_ghost_cells(const Field3D& U) {
    const auto& G = *U.g;
    GhostCells3D ghosts;
    ghosts.west.resize((G.Ny + 1) * (G.Nz + 1));
    ghosts.east = ghosts.west;
    ghosts.south.resize((G.Nx + 1) * (G.Nz + 1));
    ghosts.north = ghosts.south;
    ghosts.bottom.resize((G.Nx + 1) * (G.Ny + 1));
    ghosts.top = ghosts.bottom;

    auto idx2 = [&](int a, int b, int na) { return b * (na + 1) + a; };

    for (int j = 0; j <= G.Ny; ++j)
        for (int k = 0; k <= G.Nz; ++k) {
            ghosts.west[idx2(j, k, G.Ny)] = U.u[U.id(0, j, k)];
            ghosts.east[idx2(j, k, G.Ny)] = U.u[U.id(G.Nx, j, k)];
        }
    for (int i = 0; i <= G.Nx; ++i)
        for (int k = 0; k <= G.Nz; ++k) {
            ghosts.south[idx2(i, k, G.Nx)] = U.u[U.id(i, 0, k)];
            ghosts.north[idx2(i, k, G.Nx)] = U.u[U.id(i, G.Ny, k)];
        }
    for (int i = 0; i <= G.Nx; ++i)
        for (int j = 0; j <= G.Ny; ++j) {
            ghosts.bottom[idx2(i, j, G.Nx)] = U.u[U.id(i, j, 0)];
            ghosts.top[idx2(i, j, G.Nx)] = U.u[U.id(i, j, G.Nz)];
        }
    return ghosts;
}
