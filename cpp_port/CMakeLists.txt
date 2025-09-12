cmake_minimum_required(VERSION 3.12)
project(car_model)

# Находим pybind11. Убедитесь, что он установлен (pip install pybind11)
find_package(pybind11 CONFIG REQUIRED)

# Создаем Python модуль
pybind11_add_module(car_model SHARED bindings.cpp)

# Указываем стандарт C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)