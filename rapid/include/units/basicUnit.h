#pragma once

#include "../internal.h"

namespace rapid
{
	namespace units
	{
		class Unit
		{
		public:
			Unit()
			{}

			Unit(const std::string &name) : m_Name(name)
			{}

			inline const std::string &getName() const
			{
				return m_Name;
			}

		private:
			std::string m_Name;
		};

		class Value
		{
		public:
			Value() : m_Value(0), m_Unit("unknown")
			{}

			template<typename _Ty>
			Value(const _Ty val) : m_Value(val)
			{}

			template<typename _Ty>
			Value(const _Ty val, const Unit &unit) : m_Value(val), m_Unit(unit)
			{}

			inline double getValue() const
			{
				return m_Value;
			}

		private:
			double m_Value;
			Unit m_Unit;
		};
	}
}
