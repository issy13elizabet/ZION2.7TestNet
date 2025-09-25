# Tests for ZION cryptocurrency
cmake_minimum_required(VERSION 3.15)

# Simple test executable (placeholder for now)
add_executable(test_basic test_basic.cpp)
target_link_libraries(test_basic zion_core)

# Register tests
add_test(NAME basic_test COMMAND test_basic)
