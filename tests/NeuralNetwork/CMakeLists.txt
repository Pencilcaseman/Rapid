cmake_minimum_required (VERSION 3.8)

add_executable (NetworkTest "networkTest.cpp")

target_link_libraries(NetworkTest PRIVATE rapid)
