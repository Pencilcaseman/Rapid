#pragma once

#include "../internal.h"
#include "../array.h"
#include "activations.h"
#include "optimizers.h"
#include "layers/layerBase.h"

namespace rapid
{
	namespace neural
	{
		template<typename t = float32>
		class Network
		{
		public:
			Network() = default;

			~Network()
			{
				// Free the layers stored
				for (auto layer : m_Layers)
					delete layer;
			}

			void addLayer(layers::Layer<t> *layer)
			{
				m_Layers.emplace_back(layer);
			}

			void addLayers(const std::vector<layers::Layer<t> *> &layers)
			{
				for (const auto &layer : layers)
					addLayer(layer);
			}

			void addData(const ndarray::Array<t> &x, const ndarray::Array<t> &y)
			{
				m_Data.emplace_back(std::make_pair(x, y));
			}

			void addData(const std::vector<std::pair<ndarray::Array<t>, ndarray::Array<t>>> &data)
			{
				for (const auto &elem : data)
					addData(elem.first, elem.second);
			}

			void compile()
			{
				m_Layers[0]->construct(nullptr);
				for (uint64 i = 1; i < m_Layers.size(); i++)
					m_Layers[i]->construct(m_Layers[i - 1]);

				m_Built = true;
			}

			inline ndarray::Array<t> forward(const ndarray::Array<t> &input)
			{
				// TODO: Add more rigorous checks
				rapidAssert(input.shape[0] == m_Layers[0]->getNodes(), "Invalid input shape");

				m_Layers[0]->forward(input);
				for (uint64 i = 1; i < m_Layers.size(); i++)
					m_Layers[i]->forward(m_Layers[i - 1]->getPrevOutput());

				return m_Layers[m_Layers.size() - 1]->getPrevOutput();
			}

			inline ndarray::Array<t> backward(const ndarray::Array<t> &input, const ndarray::Array<t> &target)
			{
				auto output = forward(input);
				auto loss = target - output;

				for (int64 i = m_Layers.size() - 1; i >= 0; i--)
					loss.set(m_Layers[i]->backward(loss));

				return loss;
			}

		private:
			bool m_Built = false;

			std::vector<layers::Layer<t> *> m_Layers;
			std::vector<std::pair<ndarray::Array<t>, ndarray::Array<t>>> m_Data;
		};
	}
}
