cmake_minimum_required(VERSION 3.20)
project(ZionSimpleMiner VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific settings
if(WIN32)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
    add_definitions(-DNOMINMAX)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# Include directories
include_directories(src/mining)

# Source files
set(MINING_SOURCES
    src/mining/zion-simple-miner.cpp
)

set(MAIN_SOURCES
    src/main/zion-test-main.cpp
)

# Create executable
add_executable(zion-simple-test
    ${MINING_SOURCES}
    ${MAIN_SOURCES}
)

# Platform-specific linking
if(WIN32)
    target_link_libraries(zion-simple-test ws2_32)
else()
    target_link_libraries(zion-simple-test pthread)
endif()

# Compiler-specific options
if(MSVC)
    target_compile_options(zion-simple-test PRIVATE /W3)
else()
    target_compile_options(zion-simple-test PRIVATE -Wall -Wextra -O3)
endif()

# Set output directory
set_target_properties(zion-simple-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "✅ ZION Simple Miner configured successfully!")
message(STATUS "🎯 Target: zion-simple-test")
message(STATUS "🔧 C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "🖥️ Platform: ${CMAKE_SYSTEM_NAME}")