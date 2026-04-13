cmake_minimum_required(VERSION 3.22)
project(axon VERSION 0.1.0 LANGUAGES CXX)

# Settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force release if no build type specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(AXON_BUILD_PYTHON  "Build pybind11 Python extension" ON)
option(AXON_ENABLE_OPENMP "Enable OpenMP parallelism" ON)

# Dependencies
include(FetchContent)

# pybind11
if(AXON_BUILD_PYTHON)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v3.0.2
  )
  FetchContent_MakeAvailable(pybind11)
endif()

# OpenMP
if(AXON_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/axon
)

# Subdirectories
add_subdirectory(src/core)
add_subdirectory(src/data)
add_subdirectory(src/nn)
add_subdirectory(src/optimizers)
add_subdirectory(src/loss)

add_library(axon INTERFACE)
add_library(AXON::axon ALIAS axon)
target_link_libraries(axon INTERFACE core data nn optimizers loss)
target_include_directories(axon INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/axon)

add_executable(axon_test src/tests.cpp)
target_link_libraries(axon_test PRIVATE axon)

add_subdirectory(examples)
