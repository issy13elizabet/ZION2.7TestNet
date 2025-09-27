# ZIcmake_minimum_required(VERSION 3.16)
project(ZionAIMiner VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type detection
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# Platform detection
if(CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(ANDROID_BUILD ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(IOS_BUILD ON)
else()
    set(DESKTOP_BUILD ON)
endif()

# Feature options
option(ENABLE_GPU_MINING "Enable GPU mining support" ON)
option(ENABLE_AI_ACCELERATION "Enable AI acceleration" ON)
option(ENABLE_TESTING "Enable unit testing" OFF)
option(ENABLE_DOCS "Enable documentation generation" OFF)
option(ENABLE_DEV_TOOLS "Enable development tools" OFF)i-Platform Mining Engine - MIT Licensed
# Supports: Windows 11, Ubuntu, macOS, Android, iOS
# Copyright (c) 2025 Maitreya-ZionNet

cmake_minimum_required(VERSION 3.16)
project(ZionMiner VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform Detection
if(WIN32)
    set(PLATFORM_NAME "Windows")
    set(PLATFORM_WINDOWS TRUE)
elseif(APPLE)
    if(IOS)
        set(PLATFORM_NAME "iOS")
        set(PLATFORM_IOS TRUE)
        set(PLATFORM_MOBILE TRUE)
    else()
        set(PLATFORM_NAME "macOS")
        set(PLATFORM_MACOS TRUE)
    endif()
elseif(ANDROID)
    set(PLATFORM_NAME "Android")
    set(PLATFORM_ANDROID TRUE)
    set(PLATFORM_MOBILE TRUE)
else()
    set(PLATFORM_NAME "Linux")
    set(PLATFORM_LINUX TRUE)
endif()

message(STATUS "Building ZION Miner for ${PLATFORM_NAME}")

# Build Type Configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(MSVC)
    # Visual Studio
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /permissive-")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG /MD")
    
    # Enable AVX2 on Windows
    if(NOT PLATFORM_MOBILE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
    endif()
else()
    # GCC/Clang
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    
    # Mobile optimizations
    if(PLATFORM_MOBILE)
        set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
        if(PLATFORM_ANDROID)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
        endif()
    endif()
endif()

# Feature Detection and Options
option(ENABLE_OPENCL "Enable OpenCL GPU mining support" ON)
option(ENABLE_CUDA "Enable CUDA GPU mining support" ON)
option(ENABLE_CPU_MINING "Enable CPU mining support" ON)
option(ENABLE_MOBILE_OPTIMIZATION "Enable mobile-specific optimizations" ${PLATFORM_MOBILE})
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_BENCHMARKS "Build benchmark tools" OFF)

# Mobile platform adjustments
if(PLATFORM_MOBILE)
    set(ENABLE_CUDA OFF)  # No CUDA on mobile
    set(BUILD_TESTS OFF)
    set(BUILD_BENCHMARKS OFF)
    message(STATUS "Mobile platform detected - disabling CUDA and heavy features")
endif()

# Source Files
set(ZION_MINER_SOURCES
    zion-miner-mit.cpp
    cosmic-harmony-algo.cpp
    stratum-client.cpp
    zion-ai-mining.cpp
    auto-donate.cpp
)

set(ZION_MINER_HEADERS
    zion-miner-mit.h
    zion-ai-mining.h
    auto-donate.h
)

# Platform-specific sources
if(PLATFORM_WINDOWS)
    list(APPEND ZION_MINER_SOURCES src/platform/windows/windows_utils.cpp)
elseif(PLATFORM_LINUX)
    list(APPEND ZION_MINER_SOURCES src/platform/linux/linux_utils.cpp)
elseif(PLATFORM_MACOS)
    list(APPEND ZION_MINER_SOURCES src/platform/macos/macos_utils.cpp)
elseif(PLATFORM_ANDROID)
    list(APPEND ZION_MINER_SOURCES src/platform/android/android_utils.cpp)
elseif(PLATFORM_IOS)
    list(APPEND ZION_MINER_SOURCES src/platform/ios/ios_utils.cpp)
endif()

# Create main mining library
add_library(zion-mining-engine STATIC ${ZION_MINER_SOURCES} ${ZION_MINER_HEADERS})

target_include_directories(zion-mining-engine 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mining
)

# Preprocessor definitions
target_compile_definitions(zion-mining-engine PRIVATE
    ZION_MINER_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    ZION_MINER_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    ZION_MINER_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

if(PLATFORM_MOBILE)
    target_compile_definitions(zion-mining-engine PRIVATE ZION_MOBILE_BUILD=1)
endif()

# Platform-specific libraries and settings
if(PLATFORM_WINDOWS)
    target_link_libraries(zion-mining-engine PRIVATE ws2_32 crypt32 bcrypt)
    target_compile_definitions(zion-mining-engine PRIVATE _WIN32_WINNT=0x0601)
elseif(PLATFORM_LINUX)
    target_link_libraries(zion-mining-engine PRIVATE pthread dl)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENSSL openssl)
        if(OPENSSL_FOUND)
            target_link_libraries(zion-mining-engine PRIVATE ${OPENSSL_LIBRARIES})
            target_include_directories(zion-mining-engine PRIVATE ${OPENSSL_INCLUDE_DIRS})
            target_compile_definitions(zion-mining-engine PRIVATE HAVE_OPENSSL=1)
        endif()
    endif()
elseif(PLATFORM_MACOS)
    target_link_libraries(zion-mining-engine PRIVATE pthread)
    find_library(SECURITY_FRAMEWORK Security)
    if(SECURITY_FRAMEWORK)
        target_link_libraries(zion-mining-engine PRIVATE ${SECURITY_FRAMEWORK})
    endif()
elseif(PLATFORM_ANDROID)
    target_link_libraries(zion-mining-engine PRIVATE log)
elseif(PLATFORM_IOS)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(SECURITY_FRAMEWORK Security)
    if(FOUNDATION_FRAMEWORK AND SECURITY_FRAMEWORK)
        target_link_libraries(zion-mining-engine PRIVATE 
            ${FOUNDATION_FRAMEWORK} 
            ${SECURITY_FRAMEWORK}
        )
    endif()
endif()

# OpenCL Support
if(ENABLE_OPENCL AND NOT PLATFORM_IOS)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        target_link_libraries(zion-mining-engine PRIVATE OpenCL::OpenCL)
        target_compile_definitions(zion-mining-engine PRIVATE ENABLE_OPENCL=1)
        message(STATUS "OpenCL support enabled")
    else()
        message(WARNING "OpenCL requested but not found")
        set(ENABLE_OPENCL OFF)
    endif()
endif()

# CUDA Support
if(ENABLE_CUDA AND NOT PLATFORM_MOBILE)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(zion-mining-engine PRIVATE CUDA::cudart CUDA::cuda_driver)
        target_compile_definitions(zion-mining-engine PRIVATE ENABLE_CUDA=1)
        message(STATUS "CUDA support enabled")
    else()
        message(WARNING "CUDA requested but not found")
        set(ENABLE_CUDA OFF)
    endif()
endif()

# CPU Mining Support
if(ENABLE_CPU_MINING)
    target_compile_definitions(zion-mining-engine PRIVATE ENABLE_CPU_MINING=1)
    message(STATUS "CPU mining support enabled")
endif()

# Create executables for different platforms

# Desktop Executable
if(NOT PLATFORM_MOBILE)
    add_executable(zion-miner src/main/desktop_main.cpp)
    target_link_libraries(zion-miner PRIVATE zion-mining-engine)
    
    # Install desktop executable
    install(TARGETS zion-miner
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

# Mobile Library (for Android/iOS integration)
if(PLATFORM_MOBILE)
    add_library(zion-miner-mobile SHARED src/main/mobile_main.cpp)
    target_link_libraries(zion-miner-mobile PRIVATE zion-mining-engine)
    
    if(PLATFORM_ANDROID)
        # Android-specific configuration
        target_compile_definitions(zion-miner-mobile PRIVATE ANDROID_BUILD=1)
    elseif(PLATFORM_IOS)
        # iOS-specific configuration
        set_target_properties(zion-miner-mobile PROPERTIES
            FRAMEWORK TRUE
            FRAMEWORK_VERSION A
            MACOSX_FRAMEWORK_IDENTIFIER com.zion.miner
            PUBLIC_HEADER "${ZION_MINER_HEADERS}"
        )
    endif()
endif()

# Command Line Tool
add_executable(zion-cli src/main/cli_main.cpp)
target_link_libraries(zion-cli PRIVATE zion-mining-engine)

# Benchmark Tool (if enabled)
if(BUILD_BENCHMARKS AND NOT PLATFORM_MOBILE)
    add_executable(zion-benchmark src/tools/benchmark.cpp)
    target_link_libraries(zion-benchmark PRIVATE zion-mining-engine)
endif()

# Unit Tests (if enabled)
if(BUILD_TESTS AND NOT PLATFORM_MOBILE)
    enable_testing()
    
    add_executable(zion-tests 
        src/tests/test_main.cpp
        src/tests/test_cosmic_harmony.cpp
        src/tests/test_stratum_client.cpp
        src/tests/test_device_controller.cpp
    )
    target_link_libraries(zion-tests PRIVATE zion-mining-engine)
    
    add_test(NAME ZionMinerTests COMMAND zion-tests)
endif()

# Installation
install(TARGETS zion-mining-engine
    EXPORT ZionMinerTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/zion
)

install(FILES ${ZION_MINER_HEADERS}
    DESTINATION include/zion
)

# Install CLI tool
install(TARGETS zion-cli
    RUNTIME DESTINATION bin
)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ZionMinerConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ZionMinerConfig.cmake"
    INSTALL_DESTINATION lib/cmake/ZionMiner
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ZionMinerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ZionMinerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ZionMinerConfigVersion.cmake"
    DESTINATION lib/cmake/ZionMiner
)

export(EXPORT ZionMinerTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/ZionMinerTargets.cmake"
)

install(EXPORT ZionMinerTargets
    FILE ZionMinerTargets.cmake
    DESTINATION lib/cmake/ZionMiner
)

# Platform-specific packaging
if(PLATFORM_WINDOWS)
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "ZION Miner")
    set(CPACK_NSIS_PACKAGE_NAME "ZionMiner")
elseif(PLATFORM_MACOS)
    set(CPACK_GENERATOR "Bundle;TGZ")
    set(CPACK_BUNDLE_NAME "ZionMiner")
elseif(PLATFORM_LINUX)
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Maitreya-ZionNet")
endif()

set(CPACK_PACKAGE_NAME "ZionMiner")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "ZION Multi-Algorithm Cryptocurrency Miner")
set(CPACK_PACKAGE_VENDOR "Maitreya-ZionNet")

include(CPack)

# Include platform-specific configuration from parent directory
include(../cmake/PlatformConfig.cmake)

# Print build summary
message(STATUS "")
message(STATUS "ZION AI Miner Build Configuration:")
message(STATUS "  Platform: ${PLATFORM_NAME}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CPU Mining: ${ENABLE_CPU_MINING}")
message(STATUS "  OpenCL Support: ${ENABLE_OPENCL}")
message(STATUS "  CUDA Support: ${ENABLE_CUDA}")
message(STATUS "  AI Enhancement: ${ENABLE_AI_ACCELERATION}")
message(STATUS "  Mobile Optimization: ${ENABLE_MOBILE_OPTIMIZATION}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "  Benchmarks: ${BUILD_BENCHMARKS}")
message(STATUS "")