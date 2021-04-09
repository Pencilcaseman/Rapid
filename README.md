# Rapid

## What is it?

Rapid is a fast, general purpose C++ library which has been optimized to provide fast routines for many common functions, as well as providing helper classes to accelerate your programs.

---

## What does it offer?

Rapid provides many features, including:

* N-Dimensional array library
* Math utilities
* Neural network library
* Mathematical expression and inequality solver
* Message boxes
* Vectors (currently only 2D vector support)
* IO utilities

Rapid also bundles [`Mahi-Gui`](https://github.com/mahilab/mahi-gui) to allow advanced GUI systems to be created quickly and easily.

---

## How can you use it?

Rapid is intended to be used with CMake and its [`FetchContent`](https://cmake.org/cmake/help/v3.11/module/FetchContent.html). To extract the library and include it in your project, copy the code snippet below into your ```CMakeLists.txt```.

```cmake
include(FetchContent) 
FetchContent_Declare(rapid GIT_REPOSITORY https://github.com/Pencilcaseman/Rapid.git) 
FetchContent_MakeAvailable(rapid)


add_executable (myApp "myApp.cpp")
target_link_libraries(myApp rapid)
```

---

## Build Rapid yourself

To build rapid yourself, simply run the following commands and the project will be created and configured.

```cmd
mkdir build && cd build
cmake ..
cmake --build . --config "Release"
```
