cmake_minimum_required (VERSION 3.8)

include(FetchContent) 
FetchContent_Declare(mahi-gui GIT_REPOSITORY https://github.com/mahilab/mahi-gui.git) 
FetchContent_MakeAvailable(mahi-gui)

add_executable (RapidTests "main.cpp")

target_link_libraries(RapidTests
    PUBLIC
    rapid
	mahi::gui
)
