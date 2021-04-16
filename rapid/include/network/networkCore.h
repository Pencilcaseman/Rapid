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
		namespace utils
		{
			template<typename t>
			inline void checkValid(const ndarray::Array<t> &arr, const std::vector<uint64> &prevShape, uint64 targetNodes)
			{
				if (!arr.isInitialized())
				{
					std::string shape = "(";
					for (const auto &val : prevShape)
					{
						if (val != *(prevShape.end() - 1))
							shape += std::to_string(val) + ", ";
						else
							shape += std::to_string(val);
					}
					shape += ")";

					std::string n = std::to_string(targetNodes);

					message::RapidError("Neural Network Error", "Input shape " + shape + " is invalid. Expected shape (" + n + ") or (" + n + ", 1) or (1, " + n + ")").display();
				}
			}
		}

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
				for (uint64 i = 0; i < m_Layers.size(); i++)
				{
					for (uint64 j = 0; j < m_Layers.size(); j++)
					{
						if (i != j && m_Layers[i]->check(m_Layers[j]))
						{
							message::RapidWarning("Neural Network Warning", "Layers " + std::to_string(i) + " and " + std::to_string(j) + " share memory pointers, which may lead to issues and incorrect results").display();
						}
					}
				}

				m_Layers[0]->construct(nullptr);
				for (uint64 i = 1; i < m_Layers.size(); i++)
					m_Layers[i]->construct(m_Layers[i - 1]);

				m_Built = true;
			}

			inline ndarray::Array<t> forward(const ndarray::Array<t> &input, bool preFixed = false)
			{
				if (!preFixed)
				{
					auto fixed = validateArray(input, true);

				#ifdef RAPID_DEBUG
					utils::checkValid(fixed, input.shape, m_Layers[0]->getNodes());
				#endif

					m_Layers[0]->forward(fixed);
				}
				else
				{
					m_Layers[0]->forward(input);
				}

				for (uint64 i = 1; i < m_Layers.size(); i++)
					m_Layers[i]->forward(m_Layers[i - 1]->getPrevOutput());
				return m_Layers[m_Layers.size() - 1]->getPrevOutput();
			}

			inline ndarray::Array<t> backward(const ndarray::Array<t> &input, const ndarray::Array<t> &target)
			{
				auto fixedInput = validateArray(input, true);
				auto fixedTarget = validateArray(target, false);

			#ifdef RAPID_DEBUG
				utils::checkValid(fixedInput, input.shape, m_Layers[0]->getNodes());
				utils::checkValid(fixedTarget, target.shape, m_Layers[m_Layers.size() - 1]->getNodes());
			#endif

				auto output = forward(fixedInput, false);
				auto loss = fixedTarget - output;

				for (int64 i = m_Layers.size() - 1; i >= 0; i--)
					loss.set(m_Layers[i]->backward(loss));

				return loss;
			}

			inline ndarray::Array<t> validateArray(const ndarray::Array<t> &input, bool x = true) const
			{
				uint64 index = x ? 0 : m_Layers.size() - 1;

				if (input.shape.size() == 1)
				{
					if (input.shape[0] == m_Layers[index]->getNodes())
						return input.reshaped({AUTO, 1});
					return ndarray::Array<t>();
				}

				if (input.shape.size() == 2)
				{
					if (input.shape[0] == m_Layers[index]->getNodes() && input.shape[1] == 1)
						return input;
					if (input.shape[1] == m_Layers[index]->getNodes() && input.shape[0] == 1)
						return input.transposed();
					return ndarray::Array<t>();
				}

				return ndarray::Array<t>();
			}

		private:
			bool m_Built = false;

			std::vector<layers::Layer<t> *> m_Layers;
			std::vector<std::pair<ndarray::Array<t>, ndarray::Array<t>>> m_Data;
		};
	}
}
