add_library(rapid INTERFACE)

if (EXISTS C:/opt/openblas)
	target_include_directories(rapid INTERFACE C:/opt/openblas/include)
	target_link_directories(rapid INTERFACE C:/opt/openblas/lib)
else()
	add_compile_definitions(RAPID_NO_BLAS)
endif()

target_include_directories(rapid INTERFACE ./include)
