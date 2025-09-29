cmake_minimum_required(VERSION 3.10)

# Nastavte CMake policy pro nový Boost pouze pokud je dostupná
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)
endif()

set(VERSION "0.1")
# $Format:Packaged from commit %H%nset(COMMIT %h)%nset(REFS "%d")$

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CONFIGURATION_TYPES Debug RelWithDebInfo Release CACHE STRING "Config types")
set(CMAKE_SKIP_INSTALL_RULES ON)
set(CMAKE_SKIP_PACKAGE_ALL_DEPENDENCY ON)
set(CMAKE_SUPPRESS_REGENERATION ON)
enable_testing()

# copy CTestCustom.cmake to build dir to disable long running tests in 'make test'
if(EXISTS "${CMAKE_SOURCE_DIR}/CTestCustom.cmake")
    configure_file(${CMAKE_SOURCE_DIR}/CTestCustom.cmake ${CMAKE_BINARY_DIR})
endif()

project(CryptoNote)

include_directories(include src external external.backup "${CMAKE_BINARY_DIR}/version")
if(APPLE)
  include_directories(SYSTEM /usr/include/malloc)
  enable_language(ASM)
endif()

if(MSVC)
include_directories(src/Platform/Windows)
elseif(APPLE)
include_directories(src/Platform/OSX)
else()
include_directories(src/Platform/Linux)
endif()

set(STATIC ${MSVC} CACHE BOOL "Link libraries statically")

if(MSVC)
  add_definitions("/bigobj /MP /W3 /GS- /D_CRT_SECURE_NO_WARNINGS /wd4996 /wd4345 /D_WIN32_WINNT=0x0600 /DWIN32_LEAN_AND_MEAN /DGTEST_HAS_TR1_TUPLE=0 /D_VARIADIC_MAX=8 /D__SSE4_1__")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10485760")
  if(STATIC)
    foreach(VAR CMAKE_C_FLAGS_DEBUG CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS_RELWITHDEBINFO CMAKE_CXX_FLAGS_RELWITHDEBINFO CMAKE_C_FLAGS_RELEASE CMAKE_CXX_FLAGS_RELEASE)
      string(REPLACE "/MD" "/MT" ${VAR} "${${VAR}}")
    endforeach()
  endif()
  include_directories(SYSTEM src/platform/msc)
else()
  if (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    add_definitions("-D_DEFAULT_SOURCE -D_GNU_SOURCE")
  endif()
  set(ARCH native CACHE STRING "CPU to build for: -march value or default")
  if("${ARCH}" STREQUAL "default")
    set(ARCH_FLAG "")
  else()
    set(ARCH_FLAG "-march=${ARCH}")
  endif()
  
  # Nebudeme používat -Werror, aby nepadaly detekční testy (FindThreads apod.)
  if(APPLE)
    set(WARNINGS "-Wall -Wextra -Wpointer-arith -Wundef -Wvla -Wwrite-strings -Wno-error=extra -Wno-error=unused-function -Wno-error=deprecated-declarations -Wno-error=sign-compare -Wno-error=strict-aliasing -Wno-error=type-limits -Wno-unused-parameter -Wno-error=unused-variable -Wno-error=undef -Wno-error=uninitialized -Wno-error=unused-result")
  else()
    set(WARNINGS "-Wall -Wextra -Wpointer-arith -Wundef -Wvla -Wwrite-strings -Wno-error=extra -Wno-error=unused-function -Wno-error=deprecated-declarations -Wno-error=sign-compare -Wno-error=strict-aliasing -Wno-error=type-limits -Wno-unused-parameter -Wno-error=unused-variable -Wno-error=undef -Wno-error=uninitialized -Wno-error=unused-result")
  endif()
  
  if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(WARNINGS "${WARNINGS} -Wno-error=mismatched-tags -Wno-error=null-conversion -Wno-overloaded-shift-op-parentheses -Wno-error=shift-count-overflow -Wno-error=tautological-constant-out-of-range-compare -Wno-error=unused-private-field -Wno-error=unneeded-internal-declaration -Wno-error=unused-function")
  else()
    set(WARNINGS "${WARNINGS} -Wlogical-op -Wno-error=maybe-uninitialized -Wno-error=clobbered -Wno-error=unused-but-set-variable")
  endif()
  if(MINGW)
    set(WARNINGS "${WARNINGS} -Wno-error=unused-value")
    set(MINGW_FLAG "-DWIN32_LEAN_AND_MEAN")
    include_directories(SYSTEM src/platform/mingw)
  else()
    set(MINGW_FLAG "")
  endif()
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND NOT (CMAKE_C_COMPILER_VERSION VERSION_LESS 5.1))
    set(WARNINGS "${WARNINGS} -Wno-error=odr")
  endif()
  set(C_WARNINGS "-Waggregate-return -Wnested-externs -Wold-style-definition -Wstrict-prototypes")
  set(CXX_WARNINGS "-Wno-reorder -Wno-missing-field-initializers")
  
  # Apple Silicon (ARM64) vs Intel Mac - různé AES support a intrinsics
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # ARM64 Mac - bez -maes a SSE intrinsics
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 ${MINGW_FLAG} ${WARNINGS} ${C_WARNINGS} ${ARCH_FLAG} -DARM64_BUILD")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${MINGW_FLAG} ${WARNINGS} ${CXX_WARNINGS} ${ARCH_FLAG} -DARM64_BUILD")
    # Odeberte GCC-specific warnings pro Clang na ARM64
    string(REPLACE "-Wlogical-op" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-Werror=maybe-uninitialized" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-Werror=clobbered" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    string(REPLACE "-Wlogical-op" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Werror=maybe-uninitialized" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE "-Werror=clobbered" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    # Použij -maes pouze na x86_64; na ARM a jiných architekturách ho vynecháme
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 ${MINGW_FLAG} ${WARNINGS} ${C_WARNINGS} ${ARCH_FLAG} -maes")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${MINGW_FLAG} ${WARNINGS} ${CXX_WARNINGS} ${ARCH_FLAG} -maes")
    else()
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 ${MINGW_FLAG} ${WARNINGS} ${C_WARNINGS} ${ARCH_FLAG}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${MINGW_FLAG} ${WARNINGS} ${CXX_WARNINGS} ${ARCH_FLAG}")
    endif()
  endif()
  
  if(NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  endif()
  if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGTEST_HAS_TR1_TUPLE=0")
  endif()
  if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND NOT (CMAKE_C_COMPILER_VERSION VERSION_LESS 4.8))
    set(DEBUG_FLAGS "-g3 -Og")
  else()
    set(DEBUG_FLAGS "-g3 -O0")
  endif()
  
  # Minimální optimalizace pro macOS
  set(RELEASE_FLAGS "-O2 -DNDEBUG")
  
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${DEBUG_FLAGS}")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
  if(STATIC AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")
  endif()
endif()

# macOS threading fix (Linux necháme na FindThreads)
if(APPLE)
  set(CMAKE_THREAD_LIBS_INIT "-lpthread")
  set(CMAKE_HAVE_THREADS_LIBRARY 1)
  set(CMAKE_USE_WIN32_THREADS_INIT 0)
  set(CMAKE_USE_PTHREADS_INIT 1)
  set(THREADS_PREFER_PTHREAD_FLAG ON)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads)

# Boost configuration
set(Boost_NO_BOOST_CMAKE ON)
if(APPLE)
    set(BOOST_ROOT /opt/homebrew)
    set(BOOST_INCLUDEDIR /opt/homebrew/include)
    set(BOOST_LIBRARYDIR /opt/homebrew/lib)
    
    # Explicitní linkování Boost knihoven pro macOS
    set(Boost_system_LIBRARY /opt/homebrew/lib/libboost_system.dylib)
    set(Boost_filesystem_LIBRARY /opt/homebrew/lib/libboost_filesystem.dylib)
    set(Boost_program_options_LIBRARY /opt/homebrew/lib/libboost_program_options.dylib)
    
    # Přidejte ostatní knihovny pokud existují
    if(EXISTS "/opt/homebrew/lib/libboost_thread.dylib")
        set(Boost_thread_LIBRARY /opt/homebrew/lib/libboost_thread.dylib)
    endif()
    if(EXISTS "/opt/homebrew/lib/libboost_date_time.dylib")
        set(Boost_date_time_LIBRARY /opt/homebrew/lib/libboost_date_time.dylib)
    endif()
    if(EXISTS "/opt/homebrew/lib/libboost_chrono.dylib")
        set(Boost_chrono_LIBRARY /opt/homebrew/lib/libboost_chrono.dylib)
    endif()
    if(EXISTS "/opt/homebrew/lib/libboost_regex.dylib")
        set(Boost_regex_LIBRARY /opt/homebrew/lib/libboost_regex.dylib)
    endif()
    if(EXISTS "/opt/homebrew/lib/libboost_serialization.dylib")
        set(Boost_serialization_LIBRARY /opt/homebrew/lib/libboost_serialization.dylib)
    endif()
    
    # Manuální nastavení Boost proměnných
    set(Boost_FOUND TRUE)
    set(Boost_INCLUDE_DIRS /opt/homebrew/include)
    set(Boost_LIBRARIES 
        ${Boost_system_LIBRARY}
        ${Boost_filesystem_LIBRARY} 
        ${Boost_program_options_LIBRARY}
        ${Boost_thread_LIBRARY}
        ${Boost_date_time_LIBRARY}
        ${Boost_chrono_LIBRARY}
        ${Boost_regex_LIBRARY}
        ${Boost_serialization_LIBRARY}
    )
else()
    # Pro Linux/Windows používejte původní find_package (bez explicitního 'thread' komponentu)
    if(STATIC)
      set(Boost_USE_STATIC_LIBS ON)
      set(Boost_USE_STATIC_RUNTIME ON)
    endif()
    find_package(Boost 1.55 REQUIRED COMPONENTS system filesystem date_time chrono regex serialization program_options)
endif()

include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
if(MINGW)
  set(Boost_LIBRARIES "${Boost_LIBRARIES};ws2_32;mswsock")
elseif(APPLE)
  set(Boost_LIBRARIES "${Boost_LIBRARIES};${CMAKE_THREAD_LIBS_INIT}")
elseif(NOT MSVC)
  # Přidejte boost_thread a pthread ručně kvůli známým problémům s FindThreads na některých platformách
  set(Boost_LIBRARIES "${Boost_LIBRARIES};boost_thread;pthread")
endif()

# Version handling
set(COMMIT_ID_IN_VERSION ON CACHE BOOL "Include commit ID in version")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/version")
if (NOT COMMIT_ID_IN_VERSION)
  set(VERSION "${VERSION}-unknown")
  configure_file("src/version.h.in" "version/version.h")
  add_custom_target(version ALL)
elseif(DEFINED COMMIT)
  string(REPLACE "." "\\." VERSION_RE "${VERSION}")
  if(NOT REFS MATCHES "(\\(|, )tag: v${VERSION_RE}(\\)|, )")
    set(VERSION "${VERSION}-g${COMMIT}")
  endif()
  configure_file("src/version.h.in" "version/version.h")
  add_custom_target(version ALL)
else()
  find_package(Git QUIET)
  if(Git_FOUND OR GIT_FOUND)
    message(STATUS "Found Git: ${GIT_EXECUTABLE}")
    add_custom_target(version ALL "${CMAKE_COMMAND}" "-D" "VERSION=${VERSION}" "-D" "GIT=${GIT_EXECUTABLE}" "-D" "TO=${CMAKE_BINARY_DIR}/version/version.h" "-P" "src/version.cmake" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  else()
    message(STATUS "WARNING: Git was not found!")
    set(VERSION "${VERSION}-unknown")
    configure_file("src/version.h.in" "version/version.h")
    add_custom_target(version ALL)
  endif()
endif()

add_subdirectory(external)
add_subdirectory(src)
