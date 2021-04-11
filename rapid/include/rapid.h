#pragma once

#ifdef RAPID_CUDA
#error CUDA support is limited. For more information see "README.md". Remove this error and procede at your own risk...
#endif

#include "./internal.h"
#include "./units.h"
#include "./rapid_math.h"
#include "./array.h"

#ifdef RAPID_USE_MATRIX
#include "./matrix.h"
#endif

#include "./vector.h"
#include "./network.h"
#include "./io.h"
#include "./parser.h"

// Include Mahi-Gui because it's awesome
#include <Mahi/Gui.hpp>
#include <Mahi/Util.hpp>
