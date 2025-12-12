#pragma once
#include <cstdlib>
#include <iostream>
#include <string>

struct CLIOptions {
    std::string mode;
};

inline CLIOptions parse_cli(int argc, char** argv) {
    CLIOptions opts;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) {
            opts.mode = argv[++i];
        } else {
            std::cout << "Unknown or incomplete argument: " << arg << "\n";
            std::cout << "Usage: ./solver --mode <mode>\n";
            std::exit(1);
        }
    }
    if (opts.mode.empty()) {
        std::cout << "Usage: ./solver --mode <mode>\n";
        std::exit(1);
    }
    return opts;
}
