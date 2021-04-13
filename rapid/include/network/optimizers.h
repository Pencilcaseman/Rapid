#pragma once

#include "../internal.h"
#include "../rapid_math.h"
#include "../array.h"

namespace rapid
{
	namespace neural
	{
		namespace optim
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
					return w + m_LearningRate * dw;
				}

				inline void setParam(const std::string &name, const t val) override
				{
					if (name == "learningRate")
					{
						m_LearningRate = val;
						return;
					}

					rapidAssert(false, "'Stochastic Gradient Descent' optimizer has no parameter named '" + name + "'");
				}

				inline void setParam(const std::string &name, const ndarray::Array<t> &val) override
				{
					if (name == "learningRate")
					{
						m_LearningRate = (t) val;
						return;
					}

					rapidAssert(false, "'Stochastic Gradient Descent' optimizer has no parameter named '" + name + "'");
				}

				inline const ndarray::Array<t> getParam(const std::string &name) const override
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
			class SGDMomentum : public Optimizer<t>
			{
			public:
				SGDMomentum(t learningRate = 1e-2, t momentum = 0.9, const ndarray::Array<t> &velocity = ndarray::Array<t>())
					: m_LearningRate(learningRate), m_Momentum(momentum), m_Velocity(velocity)
				{}

				inline ndarray::Array<t> apply(const ndarray::Array<t> &w, const ndarray::Array<t> &dw) override
				{
					if (!m_Velocity.isInitialized())
						m_Velocity = ndarray::zerosLike(w);

					// Momentum update formula -- also update velocity
					// m_Velocity = m_Momentum * m_Velocity - m_LearningRate * dw;
					m_Velocity = m_LearningRate * dw - m_Momentum * m_Velocity;
					return w + m_Velocity;
				}

				inline void setParam(const std::string &name, const t val) override
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

					rapidAssert(false, "'Stochastic Gradient Descent with Momentum' optimizer has no parameter named '" + name + "'");
				}

				inline void setParam(const std::string &name, const ndarray::Array<t> &val) override
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

				inline const ndarray::Array<t> getParam(const std::string &name) const override
				{
					if (name == "learningRate")
						return ndarray::fromScalar<t>(m_LearningRate);
					if (name == "momentum")
						return ndarray::fromScalar<t>(m_Momentum);
					if (name == "velocity")
						return m_Velocity;

					rapidAssert(false, "'Stochastic Gradient Descent with Momentum' optimizer has no parameter named '" + name + "'");
					return ndarray::fromScalar<t>(INFINITY);
				}

			private:
				t m_LearningRate = 1e-2;
				t m_Momentum = 0.9;
				ndarray::Array<t> m_Velocity;
			};

			template<typename t>
			class RMSProp : public Optimizer<t>
			{
			public:
				RMSProp(t learningRate = 1e-2, t decayRate = 0.99, t epsilon = 1e-8, const ndarray::Array<t> &cache = ndarray::Array<t>()) 
					: m_LearningRate(learningRate), m_DecayRate(decayRate), m_Epsilon(epsilon), m_Cache(cache)
				{}

				inline ndarray::Array<t> apply(const ndarray::Array<t> &x, const ndarray::Array<t> &dx) override
				{
					if (!m_Cache.isInitialized())
						m_Cache.set(ndarray::zerosLike(x));

					m_Cache.set(m_DecayRate * m_Cache + (1 - m_DecayRate) * (dx * dx));
					auto nextX = x - m_LearningRate * dx / (ndarray::sqrt(m_Cache) + m_Epsilon);

					return nextX;
				}

				inline void setParam(const std::string &name, const t val) override
				{
					if (name == "learningRate")
					{
						m_LearningRate = val;
						return;
					}

					if (name == "decayRate")
					{
						m_DecayRate = val;
						return;
					}

					if (name == "epsilon")
					{
						m_Epsilon = val;
						return;
					}

					if (name == "cache")
					{
						m_Cache.fill(val);
						return;
					}

					rapidAssert(false, "'RMS Prop' optimizer has no parameter named '" + name + "'");
				}

				inline void setParam(const std::string &name, const ndarray::Array<t> &val) override
				{
					if (name == "learningRate")
					{
						m_LearningRate = (t) val;
						return;
					}

					if (name == "decayRate")
					{
						m_DecayRate = (t) val;
						return;
					}

					if (name == "epsilon")
					{
						m_Epsilon = (t) val;
						return;
					}

					if (name == "cache")
					{
						m_Cache = val;
						return;
					}

					rapidAssert(false, "'Stochastic Gradient Descent with Momentum' optimizer has no parameter named '" + name + "'");
				}

				inline const ndarray::Array<t> getParam(const std::string &name) const override
				{
					if (name == "learningRate")
						return ndarray::fromScalar<t>(m_LearningRate);
					if (name == "decayRate")
						return ndarray::fromScalar<t>(m_DecayRate);
					if (name == "m_Epsilon")
						return ndarray::fromScalar<t>(m_Epsilon);
					if (name == "cache")
						return m_Cache;

					rapidAssert(false, "'Stochastic Gradient Descent' optimizer has no parameter named '" + name + "'");
					return ndarray::fromScalar<t>(INFINITY);
				}

			private:
				t m_LearningRate = 1e-2;
				t m_DecayRate = 0.99;
				t m_Epsilon = 1e-8;
				ndarray::Array<t> m_Cache;
			};

			template<typename t>
			class ADAM : public Optimizer<t>
			{
			public:
				ADAM(t learningRate = 1e-3, t beta1 = 0.9, t beta2 = 0.999, t epsilon = 1e-8, const ndarray::Array<t> &m = ndarray::Array<t>(), const ndarray::Array<t> &v = ndarray::Array<t>(), int64 time = 0) 
					: m_LearningRate(learningRate), m_Beta1(beta1), m_Beta2(beta2), m_Epsilon(epsilon), m_M(m), m_V(v), m_Time(time)
				{}

				inline ndarray::Array<t> apply(const ndarray::Array<t> &x, const ndarray::Array<t> &dx) override
				{
					if (!m_M.isInitialized())
						m_M = ndarray::zerosLike(x);

					if (!m_V.isInitialized())
						m_V = ndarray::zerosLike(x);

					m_Time++;
					m_M = m_Beta1 * m_M + (1 - m_Beta1) * dx;
					auto mCorr = m_M / (1. - std::pow(m_Beta1, (t) m_Time));
					m_V = m_Beta2 * m_V + (1 - m_Beta2) * (dx * dx);
					auto vCorr = m_V / (1 - std::pow(m_Beta2, (t) m_Time));
					auto nextX = x - m_LearningRate * mCorr / (ndarray::sqrt(vCorr) + m_Epsilon);

					return nextX;
				}

				inline void setParam(const std::string &name, const t val) override
				{
					if (name == "learningRate")
					{
						m_LearningRate = val;
						return;
					}

					if (name == "beta1")
					{
						m_Beta1 = val;
						return;
					}

					if (name == "beta2")
					{
						m_Beta2 = val;
						return;
					}

					if (name == "epsilon")
					{
						m_Epsilon = val;
						return;
					}

					if (name == "m")
					{
						m_M.fill(val);
						return;
					}

					if (name == "v")
					{
						m_V.fill(val);
						return;
					}

					if (name == "time")
					{
						m_Time = (int64) val;
						return;
					}

					rapidAssert(false, "'ADAM' optimizer has no parameter named '" + name + "'");
				}

				inline void setParam(const std::string &name, const ndarray::Array<t> &val) override
				{
					if (name == "learningRate")
					{
						m_LearningRate = (t) val;
						return;
					}

					if (name == "beta1")
					{
						m_Beta1 = (t) val;
						return;
					}

					if (name == "beta2")
					{
						m_Beta2 = (t) val;
						return;
					}

					if (name == "epsilon")
					{
						m_Epsilon = (t) val;
						return;
					}

					if (name == "m")
					{
						m_M = val;
						return;
					}

					if (name == "v")
					{
						m_V = val;
						return;
					}

					if (name == "time")
					{
						m_Time = (int64) val;
						return;
					}

					rapidAssert(false, "'ADAM' optimizer has no parameter named '" + name + "'");
				}

				inline const ndarray::Array<t> getParam(const std::string &name) const override
				{
					if (name == "learningRate")
						return ndarray::fromScalar<t>(m_LearningRate);
					if (name == "beta1")
						return ndarray::fromScalar<t>(m_Beta1);
					if (name == "beta2")
						return ndarray::fromScalar<t>(m_Beta2);
					if (name == "epsilon")
						return ndarray::fromScalar<t>(m_Epsilon);
					if (name == "m")
						return m_M;
					if (name == "v")
						return m_V;
					if (name == "time")
						return ndarray::fromScalar<t>((t) m_Time);

					rapidAssert(false, "'ADAM' optimizer has no parameter named '" + name + "'");
					return ndarray::fromScalar<t>(INFINITY);
				}

			private:
				t m_LearningRate = 1e-3;
				t m_Beta1 = 0.9;
				t m_Beta2 = 0.999;
				t m_Epsilon = 1e-8;
				ndarray::Array<t> m_M;
				ndarray::Array<t> m_V;
				int64 m_Time = 0;
			};
		}
	}
}
