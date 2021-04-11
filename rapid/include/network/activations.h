#pragma once

#include "../internal.h"
#include "../array.h"

namespace rapid
{
	namespace neural
	{
		namespace activation
		{
			template<typename t, ndarray::ArrayLocation loc>
			ndarray::Array<t, loc> relu(const ndarray::Array<t, loc> &arr)
			{
				return ndarray::maximum(arr, 0);
			}

			template<typename t, ndarray::ArrayLocation loc>
			ndarray::Array<t, loc> tanh(const ndarray::Array<t, loc> &arr)
			{
				return ndarray::tanh(arr, 0);
			}
			
			template<typename t, ndarray::ArrayLocation loc>
			ndarray::Array<t, loc> sigmoid(const ndarray::Array<t, loc> &arr)
			{
				return 1. / (1. + ndarray::exp(-arr));
			}
		}
	}
}
