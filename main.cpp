// Climate model driver: runs legacy modes and writes VTK outputs.
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "utils/cli.hpp"
#include "utils/io.hpp"
#include "utils/stats.hpp"
#include "models/heat_models.hpp"

namespace fs = std::__fs::filesystem;

const std::string RESULTS_DIR = "Results_updated";

int main(int argc, char** argv) {
    CLIOptions opts = parse_cli(argc, argv);
    fs::create_directories(RESULTS_DIR);

    auto wall_start = std::chrono::high_resolution_clock::now();
    if (opts.mode == "diffusion_2d") {
        auto field = run_diffusion_2d_simple();
        write_vtk(RESULTS_DIR + "/diffusion_2d.vtk", field);
        report_summary(opts.mode, field);
    } else if (opts.mode == "advection_2d") {
        auto field = run_advection_2d_simple();
        write_vtk(RESULTS_DIR + "/advection_2d.vtk", field);
        report_summary(opts.mode, field);
    } else if (opts.mode == "forcing_2d") {
        auto field = run_forcing_2d_simple();
        write_vtk(RESULTS_DIR + "/forcing_2d.vtk", field);
        report_summary(opts.mode, field);
    } else if (opts.mode == "full_climate_3d") {
        auto field = run_full_climate_3d_simple();
        write_vtk(RESULTS_DIR + "/full_climate_3d.vtk", field);
        report_summary(opts.mode, field);
    } else {
        std::cout << "Unknown mode: " << opts.mode << "\n";
        std::cout << "Supported modes: diffusion_2d, advection_2d, forcing_2d, full_climate_3d\n";
        return 1;
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    std::cout << "Elapsed wall time: " << std::chrono::duration<double>(wall_end - wall_start).count() << " s\n";
    return 0;
}
