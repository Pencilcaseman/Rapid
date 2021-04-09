cmake_minimum_required (VERSION 3.8 FATAL_ERROR)

project ("rapid")

set(MODERN_CMAKE_BUILD_TESTING ON)
set(COMPILE_DEFINITIONS "")

include(CTest)

# Compilation mode defaults to CPU only code
if (NOT DEFINED RAPID_MODE)
	set(RAPID_MODE "c++")
endif()

add_subdirectory("rapid")
add_subdirectory("tests")
