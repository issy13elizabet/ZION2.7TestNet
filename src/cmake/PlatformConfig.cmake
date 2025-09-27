# ZION AI Miner - Platform-Specific CMake Configuration
# Copyright (c) 2025 Maitreya-ZionNet

# Desktop platforms (Windows, Linux, macOS)
if(DESKTOP_BUILD)
    message(STATUS "Building ZION AI Miner for Desktop platforms")
    
    # Desktop executable
    add_executable(zion-miner
        main/desktop_main.cpp
        mining/zion-miner-mit.cpp
        mining/cosmic-harmony-algo.cpp
        mining/stratum-client.cpp
        mining/zion-ai-mining.cpp
    )
    
    # Desktop-specific compiler options
    if(WIN32)
        target_compile_definitions(zion-miner PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
        target_link_libraries(zion-miner PRIVATE ws2_32 winmm crypt32)
        
        # Enable high DPI awareness on Windows
        set_target_properties(zion-miner PROPERTIES
            WIN32_EXECUTABLE TRUE
            VS_DPI_AWARE "PerMonitor"
        )
    elseif(APPLE)
        target_link_libraries(zion-miner PRIVATE "-framework CoreFoundation" "-framework IOKit")
        
        # macOS deployment target
        set_target_properties(zion-miner PROPERTIES
            MACOSX_DEPLOYMENT_TARGET "10.15"
        )
    else() # Linux
        target_link_libraries(zion-miner PRIVATE pthread dl)
    endif()
    
    # Desktop optimization flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        if(MSVC)
            target_compile_options(zion-miner PRIVATE /O2 /Ob2 /DNDEBUG /arch:AVX2)
        else()
            target_compile_options(zion-miner PRIVATE -O3 -DNDEBUG -march=native -mtune=native)
        endif()
    endif()
    
    # Install desktop executable
    install(TARGETS zion-miner
        RUNTIME DESTINATION bin
        BUNDLE DESTINATION .
    )
    
    # Desktop package configuration
    if(WIN32)
        # Windows installer
        set(CPACK_GENERATOR "NSIS")
        set(CPACK_NSIS_DISPLAY_NAME "ZION AI Miner")
        set(CPACK_NSIS_PACKAGE_NAME "ZionAIMiner")
        set(CPACK_NSIS_MODIFY_PATH ON)
        
    elseif(APPLE)
        # macOS bundle
        set(CPACK_GENERATOR "DragNDrop")
        set(CPACK_DMG_FORMAT "UDZO")
        set(CPACK_DMG_VOLUME_NAME "ZION AI Miner")
        
    else()
        # Linux packages
        set(CPACK_GENERATOR "DEB;RPM;TGZ")
        set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6, libgcc1, libstdc++6")
        set(CPACK_RPM_PACKAGE_REQUIRES "glibc, libgcc, libstdc++")
    endif()
    
endif()

# Android platform
if(ANDROID_BUILD)
    message(STATUS "Building ZION AI Miner for Android")
    
    # Android shared library
    add_library(zion-miner-android SHARED
        main/android_jni.cpp
        mining/zion-miner-mit.cpp
        mining/cosmic-harmony-algo.cpp
        mining/stratum-client.cpp
        mining/zion-ai-mining.cpp
    )
    
    # Android-specific settings
    target_compile_definitions(zion-miner-android PRIVATE ANDROID __ANDROID_API__=${ANDROID_PLATFORM_LEVEL})
    
    # Android NDK libraries
    target_link_libraries(zion-miner-android PRIVATE
        android
        log
        OpenSLES
        GLESv2
    )
    
    # Android optimization
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(zion-miner-android PRIVATE
            -O3 -DNDEBUG -fvisibility=hidden -ffunction-sections -fdata-sections
        )
        target_link_options(zion-miner-android PRIVATE
            -Wl,--gc-sections -Wl,--strip-all
        )
    endif()
    
    # Android architecture-specific optimizations
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        target_compile_options(zion-miner-android PRIVATE -march=armv8-a -mtune=cortex-a76)
    elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
        target_compile_options(zion-miner-android PRIVATE -march=armv7-a -mtune=cortex-a15 -mfpu=neon)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        target_compile_options(zion-miner-android PRIVATE -march=x86-64 -mtune=intel)
    endif()
    
    # Install Android library
    install(TARGETS zion-miner-android
        LIBRARY DESTINATION lib/${ANDROID_ABI}
    )
    
endif()

# iOS platform
if(IOS_BUILD)
    message(STATUS "Building ZION AI Miner for iOS")
    
    # iOS static library
    add_library(zion-miner-ios STATIC
        main/ios_bridge.cpp
        mining/zion-miner-mit.cpp
        mining/cosmic-harmony-algo.cpp
        mining/stratum-client.cpp
        mining/zion-ai-mining.cpp
    )
    
    # iOS-specific settings
    set_target_properties(zion-miner-ios PROPERTIES
        XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "13.0"
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
        XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH "NO"
    )
    
    # iOS frameworks
    target_link_libraries(zion-miner-ios PRIVATE
        "-framework Foundation"
        "-framework UIKit"
        "-framework Security"
        "-framework Network"
    )
    
    # iOS optimization
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(zion-miner-ios PRIVATE
            -O3 -DNDEBUG -fvisibility=hidden
        )
    endif()
    
    # iOS architecture-specific optimizations
    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        target_compile_options(zion-miner-ios PRIVATE -march=armv8-a)
    endif()
    
    # Install iOS library
    install(TARGETS zion-miner-ios
        ARCHIVE DESTINATION lib
    )
    
    # Create iOS framework
    add_custom_command(TARGET zion-miner-ios POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/ZionAIMiner.framework/Headers
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:zion-miner-ios> ${CMAKE_BINARY_DIR}/ZionAIMiner.framework/ZionAIMiner
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/src/mining/zion-ai-mining.h ${CMAKE_BINARY_DIR}/ZionAIMiner.framework/Headers/
        COMMENT "Creating iOS framework"
    )
    
endif()

# Universal build settings
if(ENABLE_GPU_MINING)
    message(STATUS "GPU mining support enabled")
    
    # OpenCL support
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "OpenCL found: ${OpenCL_VERSION_STRING}")
        if(DESKTOP_BUILD)
            target_link_libraries(zion-miner PRIVATE OpenCL::OpenCL)
        endif()
        if(ANDROID_BUILD)
            target_link_libraries(zion-miner-android PRIVATE OpenCL::OpenCL)
        endif()
        target_compile_definitions(zion-miner PRIVATE ZION_ENABLE_OPENCL)
    endif()
    
    # CUDA support (desktop only)
    if(DESKTOP_BUILD AND NOT APPLE)
        find_package(CUDA QUIET)
        if(CUDA_FOUND)
            message(STATUS "CUDA found: ${CUDA_VERSION}")
            enable_language(CUDA)
            target_compile_definitions(zion-miner PRIVATE ZION_ENABLE_CUDA)
            target_link_libraries(zion-miner PRIVATE ${CUDA_LIBRARIES})
        endif()
    endif()
endif()

# AI acceleration support
if(ENABLE_AI_ACCELERATION)
    message(STATUS "AI acceleration support enabled")
    
    # Platform-specific AI libraries
    if(DESKTOP_BUILD)
        # Desktop AI libraries (OpenBLAS, Intel MKL, etc.)
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            target_link_libraries(zion-miner PRIVATE ${BLAS_LIBRARIES})
            target_compile_definitions(zion-miner PRIVATE ZION_ENABLE_BLAS)
        endif()
    endif()
    
    if(ANDROID_BUILD)
        # Android NNAPI
        target_compile_definitions(zion-miner-android PRIVATE ZION_ENABLE_NNAPI)
    endif()
    
    if(IOS_BUILD)
        # iOS Core ML / Metal Performance Shaders
        target_link_libraries(zion-miner-ios PRIVATE
            "-framework CoreML"
            "-framework MetalPerformanceShaders"
        )
        target_compile_definitions(zion-miner-ios PRIVATE ZION_ENABLE_COREML)
    endif()
endif()

# Testing support
if(ENABLE_TESTING)
    enable_testing()
    
    # Add unit tests
    add_executable(zion-miner-tests
        tests/test_cosmic_harmony.cpp
        tests/test_ai_mining.cpp
        tests/test_stratum_client.cpp
        mining/cosmic-harmony-algo.cpp
        mining/zion-ai-mining.cpp
        mining/stratum-client.cpp
    )
    
    # Link test framework
    find_package(GTest QUIET)
    if(GTest_FOUND)
        target_link_libraries(zion-miner-tests PRIVATE GTest::GTest GTest::Main)
    endif()
    
    add_test(NAME CosmicHarmonyTests COMMAND zion-miner-tests)
endif()

# Documentation
if(ENABLE_DOCS)
    find_package(Doxygen QUIET)
    if(DOXYGEN_FOUND)
        set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/docs)
        doxygen_add_docs(docs
            ${CMAKE_SOURCE_DIR}/src
            COMMENT "Generating documentation"
        )
    endif()
endif()

# Development tools
if(ENABLE_DEV_TOOLS)
    # Static analysis
    find_program(CLANG_TIDY clang-tidy)
    if(CLANG_TIDY)
        set_target_properties(zion-miner PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY}")
    endif()
    
    # Code formatting
    find_program(CLANG_FORMAT clang-format)
    if(CLANG_FORMAT)
        add_custom_target(format
            COMMAND ${CLANG_FORMAT} -i -style=file ${CMAKE_SOURCE_DIR}/src/**/*.cpp ${CMAKE_SOURCE_DIR}/src/**/*.h
            COMMENT "Formatting source code"
        )
    endif()
endif()

# Print build summary
message(STATUS "")
message(STATUS "═══════════════════════════════════════")
message(STATUS "        ZION AI Miner Build Summary")
message(STATUS "═══════════════════════════════════════")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target Platform: ${CMAKE_SYSTEM_NAME}")
if(ANDROID_BUILD)
    message(STATUS "Android ABI: ${ANDROID_ABI}")
endif()
if(IOS_BUILD)
    message(STATUS "iOS Architectures: ${CMAKE_OSX_ARCHITECTURES}")
endif()
message(STATUS "GPU Mining: ${ENABLE_GPU_MINING}")
message(STATUS "AI Acceleration: ${ENABLE_AI_ACCELERATION}")
message(STATUS "Testing: ${ENABLE_TESTING}")
message(STATUS "═══════════════════════════════════════")
message(STATUS "")