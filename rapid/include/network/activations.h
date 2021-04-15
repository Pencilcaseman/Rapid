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

			// Base activation class
			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			class Activation
			{
			public:
				inline virtual void construct(uint64 prevNodes) = 0;

				inline virtual ndarray::Array<t, loc> f(const ndarray::Array<t, loc> &arr) const = 0;
				inline virtual ndarray::Array<t, loc> df(const ndarray::Array<t, loc> &arr) const = 0;
				inline virtual ndarray::Array<t, loc> weight(const std::vector<uint64> &shape) const = 0;
			};

			/***************/
			/* Activations */
			/***************/

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			class LeakyRelu : public Activation<t, loc>
			{
			public:
				inline void construct(uint64 prevNodes) override
				{
					m_PrevNodes = prevNodes;
				}

				ndarray::Array<t, loc> f(const ndarray::Array<t, loc> &arr) const override
				{
					return arr.mapped([](t x)
					{
						return x > 0 ? x : x * 0.2;
					});
				}

				ndarray::Array<t, loc> df(const ndarray::Array<t, loc> &arr) const override
				{
					return arr.mapped([](t x)
					{
						return x > 0 ? 1 : 0.2;
					});
				}

				ndarray::Array<t, loc> weight(const std::vector<uint64> &shape) const override
				{
					auto std = std::sqrt(2. / (t) m_PrevNodes);
					auto res = ndarray::Array<t, loc>(shape);
					res.fillRandom();
					return res * std;
				}

			private:
				uint64 m_PrevNodes = 0;
			};

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			class Relu : public Activation<t, loc>
			{
			public:
				inline void construct(uint64 prevNodes) override
				{
					m_PrevNodes = prevNodes;
				}

				ndarray::Array<t, loc> f(const ndarray::Array<t, loc> &arr) const override
				{
					return ndarray::maximum(arr, 0);
				}

				ndarray::Array<t, loc> df(const ndarray::Array<t, loc> &arr) const override
				{
					return ndarray::greater(arr, 0);
				}

				ndarray::Array<t, loc> weight(const std::vector<uint64> &shape) const override
				{
					auto std = std::sqrt(2. / (t) m_PrevNodes);
					auto res = ndarray::Array<t, loc>(shape);
					res.fillRandom();
					return res * std;
				}

			private:
				uint64 m_PrevNodes = 0;
			};

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			class Tanh : public Activation<t, loc>
			{
			public:
				inline void construct(uint64 prevNodes) override
				{
					m_PrevNodes = prevNodes;
				}

				ndarray::Array<t, loc> f(const ndarray::Array<t, loc> &arr) const override
				{
					return ndarray::tanh(arr);
				}

				ndarray::Array<t, loc> df(const ndarray::Array<t, loc> &arr) const override
				{
					return 1. - (arr * arr);
				}

				ndarray::Array<t, loc> weight(const std::vector<uint64> &shape) const override
				{
					auto lower = -1. / std::sqrt((t) m_PrevNodes);
					auto upper = 1. / std::sqrt((t) m_PrevNodes);
					auto res = ndarray::Array<t, loc>(shape);
					res.fillRandom(lower, upper);
					return lower + res * (upper - lower);
				}

			private:
				uint64 m_PrevNodes = 0;
			};

			template<typename t, ndarray::ArrayLocation loc = ndarray::CPU>
			class Sigmoid : public Activation<t, loc>
			{
			public:
				inline void construct(uint64 prevNodes) override
				{
					m_PrevNodes = prevNodes;
				}

				ndarray::Array<t, loc> f(const ndarray::Array<t, loc> &arr) const override
				{
					return 1. / (1. + ndarray::exp(-arr));
				}

				ndarray::Array<t, loc> df(const ndarray::Array<t, loc> &arr) const override
				{
					return arr * (1. - arr);
				}

				ndarray::Array<t, loc> weight(const std::vector<uint64> &shape) const override
				{
					auto lower = -1 / std::sqrt((t) m_PrevNodes);
					auto upper = 1 / std::sqrt((t) m_PrevNodes);
					auto res = ndarray::Array<t, loc>(shape);
					res.fillRandom(lower, upper);
					return lower + res * (upper - lower);
				}

			private:
				uint64 m_PrevNodes = 0;
			};
		}
	}
}
