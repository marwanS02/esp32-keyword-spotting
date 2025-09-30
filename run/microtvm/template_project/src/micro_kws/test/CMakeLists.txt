# This is the project CMakeLists.txt file for the test subproject
cmake_minimum_required(VERSION 3.5)

# Include the components directory of the main application:
#
set(EXTRA_COMPONENT_DIRS ${CMAKE_SOURCE_DIR}/../../student/)

if("${IDF_TARGET}" STREQUAL "linux")
    list(APPEND EXTRA_COMPONENT_DIRS "$ENV{IDF_PATH}/tools/mocks/esp_timer/")
    list(APPEND EXTRA_COMPONENT_DIRS "$ENV{IDF_PATH}/tools/mocks/startup/")
endif()

# Set the components to include the tests for.
# This can be overriden from CMake cache:
# - when invoking CMake directly: cmake -D TEST_COMPONENTS="xxxxx" ..
# - when using idf.py: idf.py -T xxxxx build
#
# set(TEST_COMPONENTS "testable" CACHE STRING "List of components to test")
set(TEST_COMPONENTS "posterior" CACHE STRING "List of components to test")
set(COMPONENTS main unity posterior)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(unit_test_test)
