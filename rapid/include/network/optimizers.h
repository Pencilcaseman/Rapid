#pragma once

#include "../internal.h"
#include "../rapid_math.h"
#include "../array.h"

namespace rapid
{
	namespace network
	{
		namespace optimizers
		{
			template<typename t>
			class Optimizer
			{
			public:
				inline virtual ndarray::Array<t> apply(const ndarray::Array<t> &w, const ndarray::Array<t> &dx) = 0;

				inline virtual void setParam(const std::string &name, const t val)
				{};
				inline virtual void setParam(const std::string &name, const ndarray::Array<t> &val)
				{};

				inline virtual const ndarray::Array<t> getParam(const std::string &name) const = 0;
			};

			template<typename t>
			class SGD : public Optimizer<t>
			{
			public:
				SGD(const t learningRate = 1e-2) : m_LearningRate(learningRate)
				{}

				inline ndarray::Array<t> apply(const ndarray::Array<t> &w, const ndarray::Array<t> &dw)
				{
					return m_LearningRate * dw;
				}

				inline void setParam(const std::string &name, const t val)
				{
					if (name == "learningRate")
					{
						m_LearningRate = val;
						return;
					}

					rapidAssert(false, "'Stochastic Gradient Descent' optimizer has no parameter named '" + name + "'");
				}

				inline void setParam(const std::string &name, const ndarray::Array<t> val)
				{
					if (name == "learningRate")
					{
						m_LearningRate = (t) val;
						return;
					}

					rapidAssert(false, "Stochastic Gradient Descent optimizer has no parameter named '" + name + "'");
				}

				inline const ndarray::Array<t> getParam(const std::string &name) const
				{
					if (name == "learningRate")
						return ndarray::fromScalar<t>(m_LearningRate);

					rapidAssert(false, "'Stochastic Gradient Descent' optimizer has no parameter named '" + name + "'");
					return ndarray::fromScalar<t>(INFINITY);
				}

			private:
				t m_LearningRate = 1e-2;
			};

			template<typename t>
			class SDGMomentum : public Optimizer<t>
			{
			public:
				SDGMomentum(t learningRate = 1e-2, t momentum = 0.9, const ndarray::Array<t> &velocity = ndarray::Array<t>()) : m_LearningRate(learningRate), m_Momentum(momentum), m_Velocity(velocity)
				{}

				inline ndarray::Array<t> apply(const ndarray::Array<t> &w, const ndarray::Array<t> &dw)
				{
					if (!m_Velocity.isInitialized())
						m_Velocity = ndarray::zerosLike(w);

					// Momentum update formula -- also update velocity
					m_Velocity.set(m_Momentum * m_Velocity - m_LearningRate * dw);
					return w + m_Velocity;
				}

				inline void setParam(const std::string &name, const t val)
				{
					if (name == "learningRate")
					{
						m_LearningRate = val;
						return;
					}

					if (name == "momentum")
					{
						m_Momentum = val;
						return;
					}

					if (name == "velocity")
					{
						m_Velocity.fill(val);
						return;
					}

					rapidAssert(false, "'Stochastic Gradient Descent with Momentum optimizer' has no parameter named '" + name + "'");
				}

				inline void setParam(const std::string &name, const ndarray::Array<t> val)
				{
					if (name == "learningRate")
					{
						m_LearningRate = (t) val;
						return;
					}

					if (name == "momentum")
					{
						m_Momentum = (t) val;
						return;
					}

					if (name == "velocity")
					{
						m_Velocity = val;
						return;
					}

					rapidAssert(false, "'Stochastic Gradient Descent with Momentum' optimizer has no parameter named '" + name + "'");
				}

				inline const ndarray::Array<t> getParam(const std::string &name) const
				{
					if (name == "learningRate")
						return ndarray::fromScalar<t>(m_LearningRate);
					if (name == "momentum")
						return ndarray::fromScalar<t>(m_Momentum);
					if (name == "velocity")
						return m_Velocity;

					rapidAssert(false, "Stochastic Gradient Descent optimizer has no parameter named '" + name + "'");
					return ndarray::fromScalar<t>(INFINITY);
				}

			private:
				t m_LearningRate = 1e-2;
				t m_Momentum = 0.9;
				ndarray::Array<t> m_Velocity;
			};


		}
	}
}
