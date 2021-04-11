#pragma once

#include "../internal.h"
#include "basicUnit.h"

namespace rapid
{
	namespace units
	{

		Value operator "" _um(char32_t val)
		{
			return Value((float64) val / 1000000, Unit("m"));
		}

		Value operator "" _mm(char32_t val)
		{
			return Value((float64) val / 1000, Unit("m"));
		}

		Value operator "" _cm(char32_t val)
		{
			return Value((float64) val / 100, Unit("m"));
		}

		Value operator "" _m(char32_t val)
		{
			return Value((float64) val, Unit("m"));
		}

		Value operator "" _km(char32_t val)
		{
			return Value((float64) val * 1000, Unit("m"));
		}

		/************************************************************************/
		/*                                                                      */
		/************************************************************************/

		Value operator "" _um(unsigned long long val)
		{
			return Value((float64) val / 1000000, Unit("m"));
		}

		Value operator "" _mm(unsigned long long val)
		{
			return Value((float64) val / 1000, Unit("m"));
		}

		Value operator "" _cm(unsigned long long val)
		{
			return Value((float64) val / 100, Unit("m"));
		}

		Value operator "" _m(unsigned long long val)
		{
			return Value((float64) val, Unit("m"));
		}

		Value operator "" _km(unsigned long long val)
		{
			return Value((float64) val * 1000, Unit("m"));
		}

		/************************************************************************/
		/*                                                                      */
		/************************************************************************/

		Value operator "" _um(long double val)
		{
			return Value(val / 1000000, Unit("m"));
		}

		Value operator "" _mm(long double val)
		{
			return Value(val / 1000, Unit("m"));
		}

		Value operator "" _cm(long double val)
		{
			return Value(val / 100, Unit("m"));
		}

		Value operator "" _m(long double val)
		{
			return Value(val, Unit("m"));
		}

		Value operator "" _km(long double val)
		{
			return Value(val * 1000, Unit("m"));
		}
	}
}
