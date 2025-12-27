#pragma once
#include <array>
#include <functional>

struct WeatherModel {
    double ambient_temp;
    std::function<double(double, double, double, double)> rain_rate;
    std::function<std::array<double, 3>(double, double, double, double)> wind_field;
};

WeatherModel default_weather_model();
