#pragma once

#include "../internal.h"
#include "../array.h"

template<typename t>
using activationPtr = rapid::ndarray::Array<t>(*)(const rapid::ndarray::Array<t> &);

namespace rapid
{
	namespace neural
	{
		namespace activation
		{
		#define SIGMOID(x) (1 / (1 + exp((-(x)))))
		#define TANH(x) (std::tanh((x)))
		#define RELU(x) ((x) > 0 ? (x) : 0)
		#define LEAKY_RELU(x) ((x) > 0 ? (x) : ((x) * 0.2))

		#define D_SIGMOID(y) ((y) * (1 - (y)))
		#define D_TANH(y) (1 - ((y) * (y)))
		#define D_RELU(y) ((y) > 0 ? 1 : 0)
		#define D_LEAKY_RELU(y) ((y) > 0 ? 1 : 0.2)

			/***************/
			/* Activations */
			/***************/

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			ndarray::Array<t, loc> relu(const ndarray::Array<t, loc> &arr)
			{
				return ndarray::maximum(arr, 0);
			}

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			ndarray::Array<t, loc> tanh(const ndarray::Array<t, loc> &arr)
			{
				return ndarray::tanh(arr);
			}
			
			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			ndarray::Array<t, loc> sigmoid(const ndarray::Array<t, loc> &arr)
			{
				return 1. / (1. + ndarray::exp(-arr));
			}

			/***************/
			/* Derivatives */
			/***************/

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			ndarray::Array<t, loc> reluDerivative(const ndarray::Array<t, loc> &arr)
			{
				return ndarray::greater(arr, 0);
			}

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			ndarray::Array<t, loc> tanhDerivative(const ndarray::Array<t, loc> &arr)
			{
				return 1. - (arr * arr);
			}

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			ndarray::Array<t, loc> sigmoidDerivative(const ndarray::Array<t, loc> &arr)
			{
				return arr * (1. - arr);
			}
		}
	}
}
