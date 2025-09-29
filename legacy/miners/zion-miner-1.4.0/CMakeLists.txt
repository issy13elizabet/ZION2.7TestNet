cmake_minimum_required(VERSION 3.16)
project(zion-miner-1.4.0 LANGUAGES C CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build configuration
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")

# CUDA Configuration
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "50;60;70;75;80;86;89;90")

# Find required packages
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

# OpenSSL for cryptographic functions
find_package(OpenSSL REQUIRED)

# Blake3 library (manually compiled)
find_library(BLAKE3_LIBRARY NAMES blake3 libblake3
    PATHS /usr/local/lib /usr/lib
    REQUIRED
)
find_path(BLAKE3_INCLUDE_DIR blake3.h
    PATHS /usr/local/include /usr/include
    REQUIRED
)

# Boost libraries
find_package(Boost REQUIRED COMPONENTS system filesystem thread)

# TBB for parallel processing
find_package(TBB REQUIRED)

# GSL for scientific computing
pkg_check_modules(GSL REQUIRED gsl)

# FFTW3 for Fourier transforms
pkg_check_modules(FFTW3 REQUIRED fftw3)

# Eigen3 for linear algebra
find_package(Eigen3 REQUIRED)

# CUDA and cuRAND
find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    set(ENABLE_CUDA ON)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
else()
    set(ENABLE_CUDA OFF)
    message(WARNING "CUDA not found - GPU mining will be disabled")
endif()

# OpenCL
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    set(ENABLE_OPENCL ON)
    message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
else()
    set(ENABLE_OPENCL OFF)
    message(WARNING "OpenCL not found - AMD/Intel GPU mining will be disabled")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${BLAKE3_INCLUDE_DIR})
include_directories(${OPENSSL_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${FFTW3_INCLUDE_DIRS})

if(ENABLE_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

if(ENABLE_OPENCL)
    include_directories(${OpenCL_INCLUDE_DIRS})
endif()

# Compiler definitions
# Synchronizace s root VERSION (aktualne 2.6.5) – pri zmene verze aktualizovat.
add_definitions(-DZION_VERSION="2.6.5")
add_definitions(-DZION_COSMIC_HARMONY_ENABLED)

if(ENABLE_CUDA)
    add_definitions(-DCUDA_ENABLED)
endif()

if(ENABLE_OPENCL)
    add_definitions(-DOPENCL_ENABLED)
endif()

# Source files
set(ZION_SOURCES
    zion-cosmic-harmony.cpp
    zion-gpu-miner-unified.cpp
    zion-miner-main.cpp
    stratum_client.cpp
    zion-share-submitter.cpp
)

# CUDA sources (if available)
if(ENABLE_CUDA)
    set(ZION_CUDA_SOURCES
        zion-cuda-kernel.cu
    )
endif()

# Create the executable
option(FORCE_DISABLE_CUDA "Force build without CUDA even if found" ON)

if(ENABLE_CUDA AND NOT FORCE_DISABLE_CUDA)
    add_executable(zion-miner ${ZION_SOURCES} ${ZION_CUDA_SOURCES})
    set_property(TARGET zion-miner PROPERTY CUDA_SEPARABLE_COMPILATION ON)
else()
    if(ENABLE_CUDA AND FORCE_DISABLE_CUDA)
        message(WARNING "CUDA detected but FORCE_DISABLE_CUDA=ON -> building CPU/OpenCL only miner")
    endif()
    add_executable(zion-miner ${ZION_SOURCES})
    set(ENABLE_CUDA OFF)
endif()

# Link libraries
target_link_libraries(zion-miner
    ${BLAKE3_LIBRARY}
    OpenSSL::SSL
    OpenSSL::Crypto
    ${Boost_LIBRARIES}
    TBB::tbb
    ${GSL_LIBRARIES}
    ${FFTW3_LIBRARIES}
    Eigen3::Eigen
    Threads::Threads
    m  # Math library
    dl # Dynamic loading
)

if(ENABLE_CUDA)
    target_link_libraries(zion-miner ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
endif()

if(ENABLE_OPENCL)
    target_link_libraries(zion-miner OpenCL::OpenCL)
endif()

# CUDA specific settings
if(ENABLE_CUDA)
    set_target_properties(zion-miner PROPERTIES 
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
    
    # CUDA compiler flags
    target_compile_options(zion-miner PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
            -Xptxas -O3
            -Xcompiler -O3
        >
    )
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(zion-miner PRIVATE
        -O3 -march=native -mtune=native -flto
        -funroll-loops -ffast-math
        -mavx2 -mfma -mbmi2
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(zion-miner PRIVATE
        -O3 -march=native -mtune=native -flto
        -funroll-loops -ffast-math
    )
endif()

# Link-time optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set_target_properties(zion-miner PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endif()

# Installation
install(TARGETS zion-miner
    RUNTIME DESTINATION bin
)

# Print build configuration
message(STATUS "=== ZION MINER 1.4.0 BUILD CONFIGURATION ===")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA Support: ${ENABLE_CUDA}")
if(ENABLE_CUDA)
    message(STATUS "CUDA Version: ${CUDA_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "OpenCL Support: ${ENABLE_OPENCL}")
message(STATUS "Blake3 Library: ${BLAKE3_LIBRARY}")
message(STATUS "OpenSSL Version: ${OPENSSL_VERSION}")
message(STATUS "Boost Version: ${Boost_VERSION}")
message(STATUS "==============================================")