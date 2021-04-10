#pragma once

#include "../../internal.h"
#include "../optimizers.h"

namespace rapid
{
	namespace neural
	{
		namespace layers
		{
			template<typename t>
			class Layer
			{
			public:
				inline virtual void construct(const uint64 prevNodes) = 0;

				inline virtual ndarray::Array<t> forward(const ndarray::Array<t> &x) = 0;
				inline virtual ndarray::Array<t> backward(const ndarray::Array<t> &error) = 0;

			private:
				std::string m_Type;
			};
		}
	}
}