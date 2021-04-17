cmake_minimum_required (VERSION 3.8)

add_executable (SimpleNetwork "simpleNetwork.cpp")

target_link_libraries(SimpleNetwork PRIVATE rapid)
