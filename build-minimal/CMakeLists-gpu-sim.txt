cmake_minimum_required(VERSION 3.10)
project(zion-gpu-miner)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(MSVC)
    # MSVC flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /O2 /std:c++17")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    add_definitions(-D_WIN32_WINNT=0x0601)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
else()
    # GCC/Clang flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -std=c++17")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# Find threads
find_package(Threads REQUIRED)

# Source files - GPU version
set(SOURCES
    zion-gpu-miner.cpp
)

# Create executable
add_executable(zion-gpu-miner ${SOURCES})

# Link libraries
target_link_libraries(zion-gpu-miner
    Threads::Threads
)

# Windows specific libraries
if(WIN32)
    target_link_libraries(zion-gpu-miner
        ws2_32
        iphlpapi
    )
endif()

# Set binary name
set_target_properties(zion-gpu-miner PROPERTIES OUTPUT_NAME "zion-gpu-miner")

# Installation
install(TARGETS zion-gpu-miner DESTINATION bin)

# Print configuration
message(STATUS "Building ZION GPU Miner (Simulated)")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")