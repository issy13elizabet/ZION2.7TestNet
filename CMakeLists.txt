cmake_minimum_required(VERSION 3.16)
project(ZionXMRigStyle VERSION 1.1.0)

# ================= Build Options =================
option(ZION_ENABLE_RANDOMX "Enable RandomX CPU mining" ON)
option(ZION_ENABLE_CUDA "Enable CUDA experimental GPU features" OFF)
option(ZION_ENABLE_OPENCL "Enable OpenCL experimental GPU features" OFF)
option(ZION_ENABLE_UI "Build XMRig-style UI miner" ON)
option(ZION_ENABLE_EXPERIMENTAL_COSMIC "Enable experimental Cosmic Harmony hash" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(MSVC)
    add_compile_options(/W3 /bigobj)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

include(GNUInstallDirs)

# Include directories (we will add more as we structure core lib)
include_directories(src/mining src/network src/algorithms src)

# ================= Dependencies =================
if(ZION_ENABLE_RANDOMX)
    # Attempt to locate RandomX library.
    # 1. Standard system search
    find_path(RANDOMX_INCLUDE_DIR randomx.h PATH_SUFFIXES randomx)
    find_library(RANDOMX_LIBRARY NAMES randomx)

    # 2. Environment variable RANDOMX_ROOT (user-provided build path)
    if(NOT RANDOMX_INCLUDE_DIR AND DEFINED ENV{RANDOMX_ROOT})
        if(EXISTS "$ENV{RANDOMX_ROOT}/src/randomx.h")
            set(RANDOMX_INCLUDE_DIR "$ENV{RANDOMX_ROOT}/src")
        elseif(EXISTS "$ENV{RANDOMX_ROOT}/include/randomx.h")
            set(RANDOMX_INCLUDE_DIR "$ENV{RANDOMX_ROOT}/include")
        endif()
        if(WIN32)
            if(EXISTS "$ENV{RANDOMX_ROOT}/build/Release/randomx.lib")
                set(RANDOMX_LIBRARY "$ENV{RANDOMX_ROOT}/build/Release/randomx.lib")
            elseif(EXISTS "$ENV{RANDOMX_ROOT}/build/randomx.lib")
                set(RANDOMX_LIBRARY "$ENV{RANDOMX_ROOT}/build/randomx.lib")
            endif()
        else()
            if(EXISTS "$ENV{RANDOMX_ROOT}/build/librandomx.a")
                set(RANDOMX_LIBRARY "$ENV{RANDOMX_ROOT}/build/librandomx.a")
            elseif(EXISTS "$ENV{RANDOMX_ROOT}/build/librandomx.so")
                set(RANDOMX_LIBRARY "$ENV{RANDOMX_ROOT}/build/librandomx.so")
            endif()
        endif()
        if(RANDOMX_INCLUDE_DIR AND RANDOMX_LIBRARY)
            message(STATUS "RandomX resolved via RANDOMX_ROOT env: ${RANDOMX_LIBRARY}")
        endif()
    endif()

    # 3. Vendored submodule under external/RandomX (if user added it)
    if(NOT RANDOMX_LIBRARY AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/RandomX/CMakeLists.txt")
        message(STATUS "Using vendored RandomX from external/RandomX")
        add_subdirectory(external/RandomX EXCLUDE_FROM_ALL)
        set(RANDOMX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/RandomX/src")
        set(RANDOMX_LIBRARY randomx)
    endif()

    if(RANDOMX_INCLUDE_DIR AND RANDOMX_LIBRARY)
        message(STATUS "RandomX found: ${RANDOMX_LIBRARY}")
        add_definitions(-DZION_HAVE_RANDOMX)
        include_directories(${RANDOMX_INCLUDE_DIR})
        set(RANDOMX_LIBS ${RANDOMX_LIBRARY})
    else()
        message(WARNING "RandomX library not found – CPU mining will compile but hashing disabled (stubs). Provide RandomX library (set RANDOMX_ROOT or add external/RandomX).")
    endif()
endif()

# Core mining sources
set(MINING_SOURCES
    src/mining/zion-simple-miner.cpp
    src/algorithms/zion-cosmic-harmony.cpp
)

set(GPU_SOURCES)
if(ZION_ENABLE_CUDA)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        add_definitions(-DZION_HAVE_CUDA)
        list(APPEND GPU_SOURCES src/mining/zion-gpu-miner.cpp src/mining/zion-cosmic-harmony.cu)
        message(STATUS "CUDA enabled")
    else()
        message(WARNING "CUDA requested but not found – disabling")
    endif()
endif()

if(ZION_ENABLE_OPENCL)
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
        add_definitions(-DZION_HAVE_OPENCL)
        include_directories(${OpenCL_INCLUDE_DIRS})
        list(APPEND GPU_SOURCES src/mining/zion-gpu-miner.cpp src/mining/zion-cosmic-harmony.cl)
        set(GPU_LIBRARIES ${OpenCL_LIBRARIES})
        message(STATUS "OpenCL enabled")
    else()
        message(WARNING "OpenCL requested but not found – disabling")
    endif()
endif()

set(NETWORK_SOURCES src/network/pool-connection.cpp)

# ================ Core Mining Library =================
set(CORE_MINING_SOURCES
    src/mining/core/job_manager.cpp
    src/mining/core/cpu_worker.cpp
    src/mining/core/share_submitter.cpp
    src/mining/core/hashrate_stats.cpp
    src/mining/core/mining_stats.cpp
    src/algorithms/zion-cosmic-harmony.cpp
    src/network/stratum_client.cpp
)

if(NOT ZION_ENABLE_EXPERIMENTAL_COSMIC)
    # If experimental cosmic hash disabled, we still compile file but guard usage
    add_definitions(-DZION_DISABLE_COSMIC_RUNTIME)
endif()

add_library(zion_mining_core STATIC
    ${CORE_MINING_SOURCES}
    ${NETWORK_SOURCES}
    ${GPU_SOURCES}
)

target_include_directories(zion_mining_core PUBLIC src src/mining src/network src/algorithms)

if(RANDOMX_LIBS)
    target_link_libraries(zion_mining_core PUBLIC ${RANDOMX_LIBS})
endif()

if(GPU_LIBRARIES)
    target_link_libraries(zion_mining_core PUBLIC ${GPU_LIBRARIES})
endif()

if(WIN32)
    target_link_libraries(zion_mining_core PUBLIC ws2_32)
endif()

# Try to find OpenCL
find_package(OpenCL QUIET)
if(OpenCL_FOUND)
    add_definitions(-DHAVE_OPENCL)
    set(GPU_LIBRARIES ${OpenCL_LIBRARIES})
    set(GPU_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS})
    message(STATUS "OpenCL found - GPU mining enabled")
else()
    message(STATUS "OpenCL not found - GPU mining will use simulation mode")
    set(GPU_LIBRARIES "")
    set(GPU_INCLUDE_DIRS "")
endif()

# Try to find CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    add_definitions(-DHAVE_CUDA)
    enable_language(CUDA)
    message(STATUS "CUDA found - NVIDIA GPU acceleration enabled")
endif()

if(ZION_ENABLE_UI)
    add_executable(zion-xmrig-ui src/main/zion-xmrig-simple.cpp)
    target_link_libraries(zion-xmrig-ui PRIVATE zion_mining_core)
    if(WIN32)
        target_link_libraries(zion-xmrig-ui PRIVATE winmm)
    endif()
    set_target_properties(zion-xmrig-ui PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

# Core CLI miner
add_executable(zion-core-miner src/main/zion-core-miner.cpp)
target_link_libraries(zion-core-miner PRIVATE zion_mining_core)
if(RANDOMX_LIBS)
    target_link_libraries(zion-core-miner PRIVATE ${RANDOMX_LIBS})
endif()
if(WIN32)
    target_link_libraries(zion-core-miner PRIVATE ws2_32)
endif()
set_target_properties(zion-core-miner PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Interactive UI miner with keyboard controls
add_executable(zion-core-ui src/main/zion-core-ui.cpp)
target_link_libraries(zion-core-ui PRIVATE zion_mining_core)
if(RANDOMX_LIBS)
    target_link_libraries(zion-core-ui PRIVATE ${RANDOMX_LIBS})
endif()
if(WIN32)
    target_link_libraries(zion-core-ui PRIVATE ws2_32)
endif()
set_target_properties(zion-core-ui PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Copy OpenCL kernel files to output directory
if(EXISTS ${CMAKE_SOURCE_DIR}/src/mining/zion-cosmic-harmony.cl)
    configure_file(
        ${CMAKE_SOURCE_DIR}/src/mining/zion-cosmic-harmony.cl
        ${CMAKE_BINARY_DIR}/bin/zion-cosmic-harmony.cl
        COPYONLY
    )
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "===== ZION Miner Configuration Summary =====")
message(STATUS "RandomX Enabled: ${ZION_ENABLE_RANDOMX}")
message(STATUS "CUDA Requested: ${ZION_ENABLE_CUDA} (FOUND: ${CUDA_FOUND})")
message(STATUS "OpenCL Requested: ${ZION_ENABLE_OPENCL} (FOUND: ${OpenCL_FOUND})")
message(STATUS "UI Target: ${ZION_ENABLE_UI}")
message(STATUS "Experimental Cosmic Hash: ${ZION_ENABLE_EXPERIMENTAL_COSMIC}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output Directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "=============================================")
message(STATUS "")