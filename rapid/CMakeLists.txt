add_library(rapid INTERFACE)

if (EXISTS C:/opt/openblas)
	target_include_directories(rapid INTERFACE C:/opt/openblas/include)
	target_link_directories(rapid INTERFACE C:/opt/openblas/lib)
else()
	add_compile_definitions(RAPID_NO_BLAS 1)
endif()

if (DEFINED RAPID_MODE)
	if ((RAPID_MODE STREQUAL "cuda") OR (RAPID_MODE STREQUAL "CUDA"))
		message(NOTICE "RAPID_MODE is 'CUDA' -- Compiling for CUDA support, so therefore limited graphical support")
		add_compile_definitions(RAPID_CUDA 1)	

		target_include_directories(rapid INTERFACE ./include)
	elseif((RAPID_MODE STREQUAL "cxx") OR (RAPID_MODE STREQUAL "c++") OR (RAPID_MODE STREQUAL "cpp"))
		message(NOTICE "RAPID_MODE is 'C++' -- Compiling with full functionality")
		add_compile_definitions(RAPID_CXX 1)

		target_include_directories(rapid INTERFACE ./include)

		include(FetchContent) 
		FetchContent_Declare(mahi-gui GIT_REPOSITORY https://github.com/mahilab/mahi-gui.git) 
		FetchContent_MakeAvailable(mahi-gui)

		target_link_libraries(rapid INTERFACE mahi::gui)
	else()
		message(NOTICE "RAPID_MODE '" ${RAPID_MODE} "' is not valid. Define it in CMakeLists.txt to either 'cuda' or 'c++'")
	endif()
else()
	message(FATAL_ERROR "RAPID_MODE was not defined")
endif()
