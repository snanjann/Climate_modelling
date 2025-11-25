// Shared utilities for the PDE solvers.
#pragma once

#include <stdexcept>

static inline void ensure(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
