cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project ("RapidCPP")

set(MODERN_CMAKE_BUILD_TESTING ON)
include(CTest)

add_subdirectory("rapid")
# add_subdirectory ("tests")
