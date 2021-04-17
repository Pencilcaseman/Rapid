cmake_minimum_required (VERSION 3.8)

add_executable (LessSimpleNetwork "xorNetwork.cpp")

target_link_libraries(LessSimpleNetwork PRIVATE rapid)
