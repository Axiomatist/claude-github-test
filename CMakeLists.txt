cmake_minimum_required(VERSION 3.10)
project(RungeKutta CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_runge_kutta test_runge_kutta.cpp)
target_include_directories(test_runge_kutta PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

enable_testing()
add_test(NAME RungeKuttaTests COMMAND test_runge_kutta)
