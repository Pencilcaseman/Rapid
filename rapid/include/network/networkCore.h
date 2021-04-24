#pragma once

#include "../internal.h"
#include "../array.h"
#include "activations.h"
#include "optimizers.h"
#include "layers/layerBase.h"

template<typename t>
using NetworkInput = std::unordered_map<std::string, rapid::ndarray::Array<t>>;
template<typename t>
using NetworkOutput = std::unordered_map<std::string, rapid::ndarray::Array<t>>;

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

			inline uint64 sumNodes(const std::unordered_map<std::string, uint64> &nodes)
			{
				uint64 sum = 0;
				for (const auto &node : nodes)
					sum += node.second;
				return sum;
			}

			template<typename t>
			inline void findMissing(const std::unordered_map<std::string, uint64> &target,
									const std::unordered_map<std::string, ndarray::Array<t>> &given,
									const std::string &object,
									const std::string &missing)
			{
				bool valid = true;
				std::vector<std::string> notFound;

				for (const auto &param : target)
				{
					bool currentIsValid = false;

					for (const auto &inp : given)
						if (param.first == inp.first)
							currentIsValid = true;

					if (!currentIsValid)
					{
						notFound.emplace_back(param.first);
						valid = false;
					}
				}

				if (!valid)
				{
					std::string tmp;
					for (const auto &val : notFound)
					{
						if (val != *(notFound.end() - 1))
							tmp += "\"" + val + "\", ";
						else
							tmp += "\"" + val + "\"";
					}

					message::RapidError("Neural Network Error", object + " is missing required " + missing + "(s): " + tmp + "").display();
				}
			}

			template<typename t>
			inline activation::Activation<t> *newActivation(const std::string &name)
			{
				if (name == "Sigmoid")
					return new activation::Sigmoid<t>();
				if (name == "Tanh")
					return new activation::Tanh<t>();
				if (name == "Relu")
					return new activation::Relu<t>();
				if (name == "LeakyRelu")
					return new activation::LeakyRelu<t>();
				message::RapidError("Neural Network Error", "Unknown activation function '" + name + "'").display();
			}

			template<typename t>
			inline optim::Optimizer<t> *newOptimizer(const std::string &name, const t learningRate)
			{
				if (name == "SGD")
					return new optim::SGD<t>(learningRate);
				if (name == "SGDMomentum")
					return new optim::SGDMomentum<t>(learningRate);
				if (name == "RMSProp")
					return new optim::RMSProp<t>(learningRate);
				if (name == "ADAM")
					return new optim::ADAM<t>(learningRate);
				message::RapidError("Neural Network Error", "Unknown optimizer '" + name + "'").display();
			}
		}

		template<typename t = float32>
		struct NetworkConfig
		{
			std::unordered_map<std::string, uint64> inputs;
			std::unordered_map<std::string, uint64> outputs;
			std::vector<uint64> hidden;
			std::vector<std::string> activations;
			std::vector<std::string> optimizers;
			std::vector<t> learningRates;
		};

		struct TrainConfig
		{
			uint64 batchSize;
			uint64 epochs;

			TrainConfig(uint64 batch = -1, uint64 epoch = -1)
				: batchSize(batch), epochs(epoch)
			{}
		};

		template<typename t>
		class NetVis;

		template<typename t = float32>
		class Network
		{
		public:
			friend NetVis<t>;

			Network()
			{}

			Network(const std::vector<layers::Layer<t> *> &layers) : m_Layers(layers)
			{}

			Network(const NetworkConfig<t> &config)
			{
				m_HasConfig = true;

				if (config.inputs.empty())
					message::RapidError("Neural Network Error", "Neural network must take at least one input").display();
				else if (config.inputs.size() == 1)
					m_HasNamedParams = false;
				else
					m_HasNamedParams = true;

				m_Config = config;
			}

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

			// TODO: Add checks for data size compared to network size

			void addData(const ndarray::Array<t> &x, const ndarray::Array<t> &y)
			{
				if (m_HasNamedParams)
					message::RapidError("Neural Network Error", "This network requires named parameters. Please provide them").display();

				m_Data.emplace_back(std::make_pair(std::make_pair("DefaultInput", x), std::make_pair("DefaultOutput", y)));
			}

			void addData(const std::vector<ndarray::Array<t>> &x, const std::vector<ndarray::Array<t>> &y)
			{
				rapidAssert(x.size() == y.size(), "Input data and labeled data must be the same size");

				for (uint64 i = 0; i < x.size(); i++)
				{
					std::unordered_map<std::string, ndarray::Array<t>> inputMap;
					std::unordered_map<std::string, ndarray::Array<t>> outputMap;

					inputMap["defaultInput"] = x[i];
					outputMap["defaultOutput"] = y[i];

					m_Data.emplace_back(std::make_pair(inputMap, outputMap));
				}
			}

			void addData(const std::vector<NetworkInput<t>> &x, const std::vector<NetworkOutput<t>> &y)
			{
				if (!m_HasNamedParams)
					message::RapidError("Neural Network Error",
										"This network does not accept named parameters. Please do not provide them").display();

				rapidAssert(x.size() == y.size(), "Input data and labeled data must be the same size");

				for (uint64 i = 0; i < x.size(); i++)
					m_Data.emplace_back(std::make_pair(x[i], y[i]));
			}

			inline std::pair<uint64, uint64> getBatchRange()
			{
				return {m_BatchStart, m_BatchEnd};
			}

			inline void setBatchRange(uint64 start = -1, uint64 end = -1)
			{
				m_BatchStart = start, m_BatchEnd = end;
			}

			inline void record(const std::string &name)
			{
				if (name == "loss")
					m_TrackLoss = true;
				else
					message::RapidError("Neural Network Error", "Unknown request to record '" + name + "'").display();
			}

			inline void stopRecording(const std::string &name)
			{
				if (name == "loss")
					m_TrackLoss = false;

				message::RapidError("Neural Network Error", "Unknown request to stop recording '" + name + "'").display();
			}

			inline std::vector<t> getLossRecord() const
			{
				if (m_TrackLoss)
					return m_LossRecord;
				else
					message::RapidError("Neural Network Error",
										"Network is not recording loss values, so you cannot request them").display();
				return {};
			}

			inline std::vector<t> getRecord(const std::string &name) const
			{
				if (name == "loss")
					return getLossRecord();
			}

			inline double getTrainingTime() const
			{
				if (m_Training)
					return m_TimeTotal + (TIME - m_TimeStart);
				return m_TimeTotal;
			}

			void compile()
			{
				if (m_HasConfig)
				{
					// If using a config parameter, initialize using it

					uint64 activationIndex = 0;
					uint64 optimizerIndex = 0;
					uint64 lrIndex = 0;

					uint64 activationCount = m_Config.activations.size();
					if (activationCount != 0 && activationCount != 1 && activationCount != m_Config.hidden.size() + 1)
						message::RapidError("Neural Network Error", "Invalid number of activations provided. Expected 0, 1 or "
											+ std::to_string(m_Config.hidden.size() + 1)).display();

					uint64 optimCount = m_Config.activations.size();
					if (optimCount != 0 && optimCount != 1 && optimCount != m_Config.hidden.size() + 1)
						message::RapidError("Neural Network Error", "Invalid number of optimizers. Expected 0, 1 or "
											+ std::to_string(m_Config.hidden.size() + 1)).display();

					uint64 lrCount = m_Config.learningRates.size();
					if (lrCount != 0 && lrCount != 1 && lrCount != m_Config.hidden.size() + 1)
						message::RapidError("Neural Network Error", "Invalid number of optimizers. Expected 0, 1 or "
											+ std::to_string(m_Config.hidden.size() + 1)).display();

					addLayer(new layers::Input<t>(utils::sumNodes(m_Config.inputs)));

					std::string activation, optimizer;
					t lr;

					for (const auto &nodes : m_Config.hidden)
					{
						if (activationCount < 2) activation = activationCount == 0 ? "Sigmoid" : m_Config.activations[0];
						else activation = m_Config.activations[activationIndex++];

						if (optimCount < 2) optimizer = optimCount == 0 ? "SGD" : m_Config.optimizers[0];
						else optimizer = m_Config.optimizers[optimizerIndex++];

						if (lrCount < 2) lr = lrCount == 0 ? -1 : m_Config.learningRates[0];
						else lr = m_Config.learningRates[lrIndex++];

						addLayer(new layers::Affine<t>(nodes, utils::newActivation<t>(activation), utils::newOptimizer<t>(optimizer, lr)));
					}

					if (activationCount < 2) activation = activationCount == 0 ? "Sigmoid" : m_Config.activations[0];
					else activation = m_Config.activations[activationIndex++];

					if (optimCount < 2) optimizer = optimCount == 0 ? "SGD" : m_Config.optimizers[0];
					else optimizer = m_Config.optimizers[optimizerIndex++];

					if (lrCount < 2) lr = lrCount == 0 ? -1 : m_Config.learningRates[0];
					else lr = m_Config.learningRates[lrIndex++];

					addLayer(new layers::Affine<t>(utils::sumNodes(m_Config.outputs), utils::newActivation<t>(activation), utils::newOptimizer<t>(optimizer, lr)));
				}

				for (uint64 i = 0; i < m_Layers.size(); i++)
				{
					for (uint64 j = 0; j < m_Layers.size(); j++)
					{
						if (i != j && m_Layers[i]->check(m_Layers[j]))
						{
							message::RapidWarning("Neural Network Warning",
												  "Layers " + std::to_string(i) + " and " + std::to_string(j) +
												  " share memory pointers, which may lead to issues and incorrect results").display();
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

			inline std::unordered_map<std::string, ndarray::Array<t>> forward(const std::unordered_map<std::string, ndarray::Array<t>> &inputs)
			{
			#ifdef RAPID_DEBUG
				utils::findMissing<t>(m_Config.inputs, inputs, "Feed forward", "input");
			#endif

				auto tmp = forward(constructVectorFromNames(inputs));

				std::unordered_map<std::string, ndarray::Array<t>> res;
				uint64 offset = 0;

				for (const auto &out : m_Config.outputs)
				{
					auto current = ndarray::Array<t>({out.second, 1});
					memcpy(current.dataStart, tmp.dataStart + offset, sizeof(t) * out.second);
					res[out.first] = current;

					offset += out.second;
				}

				return res;
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

				return fixedTarget - output;
			}

			inline ndarray::Array<t> backward(const std::unordered_map<std::string, ndarray::Array<t>> &inputs,
											  const std::unordered_map<std::string, ndarray::Array<t>> &targets)
			{
			#ifdef RAPID_DEBUG
				utils::findMissing<t>(m_Config.inputs, inputs, "Backpropagation", "input");
				utils::findMissing<t>(m_Config.outputs, targets, "Backpropagation", "target");
			#endif

				if (m_HasNamedParams)
					return backward(constructVectorFromNames(inputs, true), constructVectorFromNames(targets, false));

				return backward(inputs.at("defaultInput"), targets.at("defaultOutput"));
			}

			// Fit the network to the training data using provided epoch and batch size parameters
			inline void fit(const TrainConfig &config = {-1, -1})
			{
				_fit(config);
			}

			// Fit the network to the training data using provided epoch and batch size parameters
			inline void fit(uint64 batchSize = -1, uint64 epochs = -1)
			{
				_fit({batchSize, epochs});
			}

			inline ndarray::Array<t> validateArray(const ndarray::Array<t> &input, bool x = true, uint64 nodes = 0) const
			{
				uint64 index = x ? 0 : m_Layers.size() - 1;

				if (input.shape.size() == 1)
				{
					if (input.shape[0] == nodes == 0 ? m_Layers[index]->getNodes() : nodes)
						return input.reshaped({AUTO, 1});
					return ndarray::Array<t>();
				}

				if (input.shape.size() == 2)
				{
					if (input.shape[0] == nodes == 0 ? m_Layers[index]->getNodes() : nodes && input.shape[1] == 1)
						return input;
					if (input.shape[1] == nodes == 0 ? m_Layers[index]->getNodes() : nodes && input.shape[0] == 1)
						return input.transposed();
					return ndarray::Array<t>();
				}

				return ndarray::Array<t>();
			}

			inline ndarray::Array<t> constructVectorFromNames(const std::unordered_map<std::string, ndarray::Array<t>> &nodes, bool input = true)
			{
				std::unordered_map<std::string, uint64> &params = input ? m_Config.inputs : m_Config.outputs;

				auto res = ndarray::Array<t>({utils::sumNodes(params), 1});
				uint64 offset = 0;

				for (const auto &param : params)
				{
					auto fixed = validateArray(nodes.at(param.first), false, param.second);

				#ifdef RAPID_DEBUG
					if (param.second != fixed.shape[0])
						message::RapidError("Neural Network Error", "Input '" + param.first + "' expected " +
											std::to_string(param.second) + " nodes, but received " + std::to_string(fixed.shape[0])).display();
				#endif

					memcpy(res.dataStart + offset, fixed.dataStart, sizeof(t) * param.second);

					offset += param.second;
				}

				return res;
			}

		private:
			inline void _fit(const TrainConfig &config)
			{
				m_TimeStart = TIME;

				m_TrainConfig = config;
				m_Training = true;

				uint64 batchStart, batchEnd;

				if (m_BatchStart != -1) batchStart = math::min(m_BatchStart, m_Data.size() - 1);
				else batchStart = 0;

				if (m_BatchEnd != -1) batchEnd = math::min(m_BatchEnd, m_Data.size());
				else batchEnd = m_Data.size();

				uint64 batchSize = batchEnd - batchStart;

				if (config.epochs == -1)
					message::RapidError("Neural Network Error", "Please specify a number of training epochs").display();

				for (; m_Epoch < config.epochs; m_Epoch++)
				{
					std::shuffle(m_Data.begin(), m_Data.end(), m_RandomGenerator);

					ndarray::Array<t> totalLoss = ndarray::zeros<t>({m_Layers[m_Layers.size() - 1]->getNodes(), 1});

					while (batchEnd < m_Data.size() + 1)
					{
						for (uint64 batch = batchStart; batch < batchEnd; batch++)
						{
							if (!m_Training)
								goto finish;

							auto loss = backward(m_Data[batch].first, m_Data[batch].second);

							if (m_TrackLoss)
								totalLoss += loss;
						}

						batchStart += batchSize;
						batchEnd += batchSize;
						m_BatchNum++;
					}

					if (m_TrackLoss)
					{
						auto meanAvg = ndarray::mean(totalLoss / (t) (batchEnd - batchStart));
						m_LossRecord.emplace_back(meanAvg * meanAvg);
					}

					batchStart = 0;
					batchEnd = batchSize;
					m_BatchNum = 0;
				}

			finish:
				m_Training = false;
				m_TimeTotal += TIME - m_TimeStart;
			}

		private:
			bool m_Built = false;
			bool m_HasNamedParams = false;

			bool m_HasConfig = false;
			NetworkConfig<t> m_Config;
			TrainConfig m_TrainConfig;

			std::mt19937 m_RandomGenerator = std::mt19937();

			std::vector<layers::Layer<t> *> m_Layers;
			std::vector<std::pair<std::unordered_map<std::string, ndarray::Array<t>>, std::unordered_map<std::string, ndarray::Array<t>>>> m_Data;

			uint64 m_BatchStart = -1, m_BatchEnd = -1;
			uint64 m_BatchNum = 0, m_Epoch = 0;

			bool m_TrackLoss = false;
			std::vector<t> m_LossRecord;

			double m_TimeStart = 0;
			double m_TimeTotal = 0;

			bool m_Training = false;
			bool m_StatisticsOpen = true;
		};
	}
}
