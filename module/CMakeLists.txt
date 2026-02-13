cmake_minimum_required(VERSION 3.10)
project(flash)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -pthread")
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -pthread")


# DEPS
set(EIGEN3_DIR "/usr/local/share/eigen3")
set(PCL_DIR "/usr/local/share/pcl-1.15")
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)


# CLIPPER
set(CLIPPER_DIR "${CMAKE_CURRENT_BINARY_DIR}/clipper-download" CACHE INTERNAL "CLIPPER build dir" FORCE)
set(CLIPPER_BUILD_BINDINGS_PYTHON OFF CACHE BOOL "Build Python bindings")
set(CLIPPER_BUILD_BINDINGS_MATLAB OFF CACHE BOOL "Build MATLAB bindings")
set(CLIPPER_BUILD_TESTS OFF CACHE BOOL "Build testsuite")
set(CLIPPER_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks")
set(CLIPPER_ENABLE_MKL OFF CACHE BOOL "Use MKL with Eigen")
set(CLIPPER_ENABLE_BLAS OFF CACHE BOOL "Use BLAS with Eigen") # apt install libopenblas-dev
set(CLIPPER_ENABLE_MAXCLIQUE ON CACHE BOOL "Use PMC lib for solveAsMaximumClique")
set(CLIPPER_ENABLE_SCS_SDR ON CACHE BOOL "Use SCS lib for solveAsMSRCSDR")
configure_file(cmake/clipper.cmake.in ${CLIPPER_DIR}/CMakeLists.txt IMMEDIATE @ONLY)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" . WORKING_DIRECTORY ${CLIPPER_DIR})
execute_process(COMMAND "${CMAKE_COMMAND}" --build . WORKING_DIRECTORY ${CLIPPER_DIR})
add_subdirectory(${CLIPPER_DIR}/src ${CLIPPER_DIR}/build)


# FLASH
set(FLASH_INCLUDE_DIRS include include/flash include/flash/external)
set(FLASH_SOURCES include/flash/flash.cpp include/flash/manager.cpp include/flash/align.cpp include/flash/external/hungarian.cpp)


add_executable(kitti  src/kitti.cpp  ${FLASH_SOURCES})
target_include_directories(kitti PRIVATE ${FLASH_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
target_link_libraries(kitti  Eigen3::Eigen ${PCL_LIBRARIES} yaml-cpp OpenMP::OpenMP_CXX clipper)

