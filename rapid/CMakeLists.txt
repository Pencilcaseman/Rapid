# Add the Rapid library as an interface
add_library(rapid INTERFACE)

# Check if OpenBLAS is installed in this location (the usual location)
# This can be changed easily by users who require a different file location
if (EXISTS C:/opt/openblas)
	target_include_directories(rapid INTERFACE C:/opt/openblas/include)
	target_link_directories(rapid INTERFACE C:/opt/openblas/lib)
else()
	add_compile_definitions(RAPID_NO_BLAS)
endif()

# Set Rapid to include the "include" directory
target_include_directories(rapid INTERFACE include)

# Fetch Mahi-Gui from GitHub
include(FetchContent) 
FetchContent_Declare(mahi-gui GIT_REPOSITORY https://github.com/mahilab/mahi-gui.git) 
FetchContent_MakeAvailable(mahi-gui)

# Link Mahi-Gui
target_link_libraries(rapid INTERFACE mahi::gui)

# Find and link against OpenMP if possible
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(rapid INTERFACE OpenMP::OpenMP_CXX)
endif()
