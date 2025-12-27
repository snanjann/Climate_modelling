#pragma once
#include <string>

struct CLIOptions {
    std::string mode;
};

CLIOptions parse_cli(int argc, char** argv);
