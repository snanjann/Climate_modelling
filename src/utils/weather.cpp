#include "weather.hpp"
#include <cmath>

WeatherModel default_weather_model() {
    WeatherModel weather;
    weather.ambient_temp = 293.15; // 20 C
    weather.rain_rate = [](double x, double y, double z, double t) {
        return 0.01 + 0.005 * std::sin(2.0 * M_PI * x + 0.5 * t)
               + 0.002 * std::cos(2.0 * M_PI * y + z + 0.1 * t);
    };
    weather.wind_field = [](double x, double y, double z, double t) {
        double u = 0.4 + 0.1 * std::cos(2.0 * M_PI * y + 0.2 * t);
        double v = 0.2 * std::sin(2.0 * M_PI * x + 0.3 * t);
        double w = 0.1 * std::cos(2.0 * M_PI * z - 0.1 * t);
        return std::array<double, 3>{u, v, w};
    };
    return weather;
}
