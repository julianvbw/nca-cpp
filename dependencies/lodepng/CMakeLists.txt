cmake_minimum_required(VERSION 3.5)
project(Lodepng)

add_library(lodepng include/lodepng.h src/lodepng.cpp)
target_include_directories(lodepng PUBLIC include/)