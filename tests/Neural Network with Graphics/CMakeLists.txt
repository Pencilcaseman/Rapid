cmake_minimum_required (VERSION 3.8)

add_executable (GraphicalNetwork "graphicalNetwork.cpp")

target_link_libraries(GraphicalNetwork PRIVATE rapid)
