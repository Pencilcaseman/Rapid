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
			class Affine
			{
			public:
				Affine(const int64 nodes, optimizers::Optimizer<t> *optimizer) : m_Nodes(nodes), m_Optimizer(optimizer)
				{}

				inline void construct(const uint64 prevNodes) override
				{
					// Construct the network to be the correct shape
					m_W = ndarray::Array<t>({m_Nodes, prevNodes});
					m_B = ndarray::Array<t>({m_Nodes, 1});
				}

				inline ndarray::Array<t> forward(const ndarray::Array<t> &x) override
				{
					rapidAssert()
				}

			private:
				uint64 m_Nodes;

				ndarray::Array<t> m_W;
				ndarray::Array<t> m_B;
				optimizers::Optimizer<t> *m_Optimizer;
			};
		}
	}
}