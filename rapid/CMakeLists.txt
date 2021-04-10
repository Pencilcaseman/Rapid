add_library(rapid INTERFACE)

if (EXISTS C:/opt/openblas)
	target_include_directories(rapid INTERFACE C:/opt/openblas/include)
	target_link_directories(rapid INTERFACE C:/opt/openblas/lib)
else()
	add_compile_definitions(RAPID_NO_BLAS 1)
endif()


add_compile_definitions(RAPID_CXX 1)

target_include_directories(rapid INTERFACE ./include)

include(FetchContent) 
FetchContent_Declare(mahi-gui GIT_REPOSITORY https://github.com/mahilab/mahi-gui.git) 
FetchContent_MakeAvailable(mahi-gui)

target_link_libraries(rapid INTERFACE mahi::gui)
