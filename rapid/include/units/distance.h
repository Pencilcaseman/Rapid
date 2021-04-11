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

		Value operator "" _um(size_t val)
		{
			return Value((float64) val / 1000000, Unit("m"));
		}

		Value operator "" _mm(size_t val)
		{
			return Value((float64) val / 1000, Unit("m"));
		}

		Value operator "" _cm(size_t val)
		{
			return Value((float64) val / 100, Unit("m"));
		}

		Value operator "" _m(size_t val)
		{
			return Value((float64) val, Unit("m"));
		}

		Value operator "" _km(size_t val)
		{
			return Value((float64) val * 1000, Unit("m"));
		}

		/************************************************************************/
		/*                                                                      */
		/************************************************************************/

		Value operator "" _um(long float64 val)
		{
			return Value(val / 1000000, Unit("m"));
		}

		Value operator "" _mm(long float64 val)
		{
			return Value(val / 1000, Unit("m"));
		}

		Value operator "" _cm(long float64 val)
		{
			return Value(val / 100, Unit("m"));
		}

		Value operator "" _m(long float64 val)
		{
			return Value(val, Unit("m"));
		}

		Value operator "" _km(long float64 val)
		{
			return Value(val * 1000, Unit("m"));
		}
	}
}
