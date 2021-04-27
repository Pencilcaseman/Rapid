message("CMake building Rapid")

# Add the Rapid library as an interface
add_library(rapid INTERFACE)

# Check if OpenBLAS is installed in this location (the usual location)
# This can be changed easily by users who require a different file location
if (EXISTS C:/opt/openblas)
	message("Rapid found OpenBLAS at C:/opt/openblas. Compiling with BLAS support")

	target_include_directories(rapid INTERFACE C:/opt/openblas/include)
	target_link_directories(rapid INTERFACE C:/opt/openblas/lib)
	target_compile_definitions(rapid INTERFACE -DRAPID_HAS_BLAS)
else()
	message("Rapid did not find OpenBLAS. Compiling without BLAS support")

	target_compile_definitions(rapid INTERFACE -DRAPID_NO_BLAS=true)
endif()

# Set Rapid to include the "include" directory
target_include_directories(rapid INTERFACE include)

# Fetch Mahi-Gui from GitHub
message("Fetching Mahi-Gui")
include(FetchContent) 
FetchContent_Declare(mahi-gui GIT_REPOSITORY https://github.com/mahilab/mahi-gui.git) 
FetchContent_MakeAvailable(mahi-gui)

# Link Mahi-Gui
target_link_libraries(rapid INTERFACE mahi::gui)

# Find and link against OpenMP if possible
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	message("Rapid found OpenMP. Compiling with OpenMP support")

    target_link_libraries(rapid INTERFACE OpenMP::OpenMP_CXX)
	target_compile_definitions(rapid INTERFACE -DRAPID_HAS_OMP)
else()
	message("Rapid did not find OpenMP. Compiling without OpenMP support")

	target_compile_definitions(rapid INTERFACE -DRAPID_NO_OMP)
endif()

# If compiling for release, enable all optimizations
if (${CMAKE_CXX_COMPILER_ID} EQUAL "MSVC")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()
