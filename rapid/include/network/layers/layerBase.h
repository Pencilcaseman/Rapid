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
				inline virtual void construct(Layer<t> *prevLayer) = 0;

				inline virtual ndarray::Array<t> forward(const ndarray::Array<t> &x) = 0;
				inline virtual ndarray::Array<t> backward(const ndarray::Array<t> &error) = 0;

				inline virtual uint64 getNodes() const = 0;
				inline virtual optim::Optimizer<t> *getOptimizer() const = 0;

				inline virtual ndarray::Array<t> getPrevOutput() const = 0;

			private:
				std::string m_Type = "none";
				ndarray::Array<t> m_PrevOutput;
			};

			template<typename t>
			class Input : public Layer<t>
			{
			public:
				Input(const uint64 nodes) : m_Nodes(nodes), m_Type("input")
				{}

				inline void construct(Layer<t> *prevLayer)
				{
					m_PrevOutput = ndarray::Array<t>({m_Nodes, 1});
				}

				inline ndarray::Array<t> forward(const ndarray::Array<t> &x) override
				{
					m_PrevOutput = x;
					return x;
				}

				inline ndarray::Array<t> backward(const ndarray::Array<t> &error) override
				{
					return error;
				}

				inline uint64 getNodes() const override
				{
					return m_Nodes;
				}

				inline optim::Optimizer<t> *getOptimizer() const override
				{
					return nullptr;
				}
			
				inline ndarray::Array<t> getPrevOutput() const override
				{
					return m_PrevOutput;
				}

			private:
				std::string m_Type;
				uint64 m_Nodes;
				ndarray::Array<t> m_PrevOutput;
			};
		}
	}
}
