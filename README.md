<p align="center">
<img src="https://github.com/Pencilcaseman/Rapid/blob/master/misc/RapidLogo.png" width="800"> 
</p>

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

To then include Rapid in your C++ project, use the following:
```c++
#include <rapid.h>
```

---

## Compile Options

Rapid provides many options to customize how it operates. These settings are initialized by CMake based on your system, though you can change them if you wish.

To enable the option, simply put the following at the top of your program:

```c++
#define RAPID_SOME_OPTION

// It is very important you define the option
// before including rapid, otherwise it will
// not take effect and will most likely cause
// very irritating problems...
#include <rapid.h>
```

Option        | Effect | Default
------------- | ------ | -------
 ```RAPID_NO_BLAS``` | Stops Rapid from utilising OpenBLAS for array operations | Only enabled if the folder ```C:/opt/OpenBLAS``` is found on the system
 ```RAPID_NO_AMP```  | Stops Rapid from utilising Microsoft AMP for array operations | Only enabled if compiling with MSVC on Windows and OpenBLAS is not being used
 ```RAPID_NO_OMP```  | Stops Rapid from utilising OpenMP | Only enabled if CMake finds OpenMP support at build time

---

## Build Rapid yourself

To build rapid yourself, simply run the following commands and the project will be created and configured.

```cmd
mkdir build && cd build
cmake ..
cmake --build . --config "Release"
```

Rapid builds successfully on Windows and MacOS, and will hopefully work on Linux too, so you can use it crossplatform without issues.

---

## Does Rapid work with CUDA?

Unfortunately, due to the way Mahi-Gui, CMake and NVCC work, it is not possible to build Rapid with support for CUDA, though the raw header files will support CUDA, so long as you ```#define RAPID_CUDA``` before you include Rapid. Due to the way Mahi-Gui is built, it will not build when compiling with the NVCC CUDA compiler, so the graphical side of Rapid will not function.

Most notably, the N-Dimensional Array library has nearly full CUDA support, and most calculations can be performed on the GPU, and casting between CPU and GPU arrays is possible.

*If anyone is aware of a workaround for this, please create a pull request or start a discussion; it would be greatly appreciated.*
