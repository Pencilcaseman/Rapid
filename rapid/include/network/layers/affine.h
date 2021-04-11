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
			class Affine : public Layer<t>
			{
			public:
				Affine(const int64 nodes, const std::pair<activationPtr<t>, activationPtr<t>> &adPair, optim::Optimizer<t> *optimizer)
					: m_Nodes(nodes), m_Activation(adPair.first), m_Derivative(adPair.second), m_Optimizer(optimizer), m_Type("affine")
				{}

				inline void construct(Layer<t> *prevLayer) override
				{
					m_PrevLayer = prevLayer;

					// Construct the network to be the correct shape
					m_W = ndarray::Array<t>({m_Nodes, m_PrevLayer->getNodes()});
					m_B = ndarray::Array<t>({m_Nodes, 1});
					m_PrevOutput = ndarray::Array<t>({m_Nodes, 1});

					m_W.fillRandom();
					m_B.fillRandom();
				}

				inline ndarray::Array<t> forward(const ndarray::Array<t> &x) override
				{
					rapidAssert(x.shape[0] == m_W.shape[1], "Cannot compute forward feed on data with " +
								std::to_string(x.shape[0]) + " nodes. Expected " + std::to_string(m_W.shape[1]) + ".");

					m_PrevOutput = m_Activation(m_W.dot(x) + m_B);
					return m_PrevOutput;
				}

				inline ndarray::Array<t> backward(const ndarray::Array<t> &error) override
				{
					// Calculate the weight gradient and adjust the weight
					// and bias for the layer accordingly. The weight update
					// is controlled by the optimizer, while the bias is
					// updated by adding the gradients

					auto gradient = m_Derivative(m_PrevOutput) * error;
					auto transposed = m_PrevLayer->getPrevOutput().transposed();
					auto dx = gradient.dot(transposed);
					m_W = m_Optimizer->apply(m_W, dx);
					m_B += gradient * m_Optimizer->getParam("learningRate");

					// Return the error to be used by earlier layers
					return m_W.transposed().dot(error);
				}

				inline uint64 getNodes() const override
				{
					return m_Nodes;
				}

				inline optim::Optimizer<t> *getOptimizer() const
				{
					return m_Optimizer;
				}

				inline ndarray::Array<t> getPrevOutput() const override
				{
					return m_PrevOutput;
				}

			private:
				std::string m_Type;
				uint64 m_Nodes;

				ndarray::Array<t> m_W;
				ndarray::Array<t> m_B;
				ndarray::Array<t> m_PrevOutput;

				optim::Optimizer<t> *m_Optimizer = nullptr;
				Layer<t> *m_PrevLayer = nullptr;

				activationPtr<t> m_Activation = nullptr;
				activationPtr<t> m_Derivative = nullptr;
			};
		}
	}
}
