#pragma once

#include "internal.h"

namespace rapid
{
	namespace math
	{
		constexpr float64 pi = 3.141592653589793238462643383279502884197169399375105820974944592307816406286;
		constexpr float64 twoPi = 6.283185307179586476925286766559005768394338798750211641949889184615632812572;
		constexpr float64 halfPi = 1.570796326794896619231321691639751442098584699687552910487472296153908203143;
		constexpr float64 e = 2.718281828459045235360287471352662497757247093699959574966967627724076630353;
		constexpr float64 sqrt2 = 1.414213562373095048801688724209698078569671875376948073176679737990732478;
		constexpr float64 sqrt3 = 1.7320508075688772935274463415058723669428052538103806280558069794519330169;
		constexpr float64 sqrt5 = 2.2360679774997896964091736687312762354406183596115257242708972454105209256378;

		template<typename T>
		inline T &&min(T &&val)
		{
			return std::forward<T>(val);
		}

		template<typename T0, typename T1, typename... Ts>
		inline auto min(T0 &&val1, T1 &&val2, Ts &&... vs)
		{
			return (val1 < val2) ?
				min(val1, std::forward<Ts>(vs)...) :
				min(val2, std::forward<Ts>(vs)...);
		}

		template<typename T>
		inline T &&max(T &&val)
		{
			return std::forward<T>(val);
		}

		template<typename T0, typename T1, typename... Ts>
		inline auto max(T0 &&val1, T1 &&val2, Ts &&... vs)
		{
			return (val1 > val2) ?
				max(val1, std::forward<Ts>(vs)...) :
				max(val2, std::forward<Ts>(vs)...);
		}

		template<typename t>
		inline t abs(t a)
		{
			if (a < 0)
				return -a;
			return a;
		}

		template<>
		inline unsigned int abs(unsigned int a)
		{
			return a;
		}

		template<>
		inline unsigned long long abs(unsigned long long a)
		{
			return a;
		}

		template<typename v, typename s, typename e, typename ss, typename ee>
		inline typename std::common_type<v, s, e, ss, ee>::type map(v n, s start1, e stop1, ss start2, ee stop2)
		{
			using _Ty = typename std::common_type<v, s, e, ss, ee>::type;
			return (_Ty) start2 + ((_Ty) stop2 - (_Ty) start2) * (((_Ty) n - (_Ty) start1) / ((_Ty) stop1 - (_Ty) start1));
		}

		inline float64 dist(float64 x1, float64 y1, float64 x2, float64 y2)
		{
			auto p1 = x2 + x1;
			auto p2 = y2 + y1;
			return std::sqrt(p1 * p1 + p2 * p2);
		}

		inline long floor(float64 val)
		{
			if (val > 0)
			{
				return (long) val;
			}
			else
			{
				return (long) (val + 1);
			}
		}

		template<typename type, typename std::enable_if<std::is_floating_point<type>::value, int>::type = 0>
		inline type random(const type &min, const type &max)
		{
			// Random floating point value in range [min, max)

			static std::uniform_real_distribution<type> distribution(0., 1.);
			static std::mt19937 generator(TIME * 1000000);
			return min + (max - min) * distribution(generator);
		}

		template<typename type, typename std::enable_if<std::is_integral<type>::value, int>::type = 0>
		inline type random(const type &min, const type &max)
		{
			// Random integral value in range [min, max]

			static std::uniform_real_distribution<float64> distribution(0., 1.);
			static std::mt19937 generator(TIME * 1000000);
			return (type) random((float64) min, (float64) max + 1);
		}

		template<typename type>
		inline type clamp(const type &x, const type &min, const type &max)
		{
			if (x < min) return min;
			if (x > max) return max;
			return x;
		}

		template<typename type, typename std::enable_if<std::is_signed<type>::value, int>::type = 0>
		inline type clamp(const type &x, const type &val)
		{
			if (x < -val && std::is_signed<type>::value) return -val;
			if (x > val) return val;
			return x;
		}

		template<typename type, typename std::enable_if<std::is_unsigned<type>::value, int>::type = 0>
		inline type clamp(const type &x, const type &val)
		{
			if (x > val) return val;
			return x;
		}

		template<typename t>
		inline t roundUp(const t &numToRound, const t &multiple)
		{
			if (multiple == 0)
				return numToRound;

			t remainder = abs(numToRound) % multiple;
			if (remainder == 0)
				return numToRound;

			if (numToRound < 0)
				return -(abs(numToRound) - remainder);
			else
				return numToRound + multiple - remainder;
		}

		template<>
		inline unsigned int roundUp(const unsigned int &numToRound, const unsigned int &multiple)
		{
			if (multiple == 0)
				return numToRound;

			unsigned int remainder = numToRound % multiple;
			if (remainder == 0)
				return numToRound;
			return numToRound + multiple - remainder;
		}

		template<>
		inline unsigned long long roundUp(const unsigned long long &numToRound, const unsigned long long &multiple)
		{
			if (multiple == 0)
				return numToRound;

			auto remainder = numToRound % multiple;
			if (remainder == 0)
				return numToRound;
			return numToRound + multiple - remainder;
		}

		template<>
		inline float32 roundUp(const float32 &numToRound, const float32 &multiple)
		{
			if (multiple == 0)
				return numToRound;

			float32 remainder = fmod(fabs(numToRound), multiple);
			if (remainder == 0)
				return numToRound;

			if (numToRound < 0)
				return -(fabs(numToRound) - remainder);
			else
				return numToRound + multiple - remainder;
		}

		template<>
		inline float64 roundUp(const float64 &numToRound, const float64 &multiple)
		{
			if (multiple == 0)
				return numToRound;

			float64 remainder = fmod(fabs(numToRound), multiple);
			if (remainder == 0)
				return numToRound;

			if (numToRound < 0)
				return -(fabs(numToRound) - remainder);
			else
				return numToRound + multiple - remainder;
		}

		template<typename t>
		inline t round(const t &numToRound, uint64 dp = 0)
		{
			auto num = abs(numToRound);

			t remainder = fmod(abs(num), 1. * pow(10, -((t) dp)));
			if (remainder == 0)
				return num * (numToRound > 0 ? 1 : -1);

			if (remainder < 0.4999999999 * pow(10, -((t) dp)))
				return (num - remainder) * (numToRound > 0 ? 1 : -1);

			return (num + (1. * pow(10, -((t) dp))) - remainder) * (numToRound > 0 ? 1 : -1);
		}

		/// <summary>
		/// Calcualate the product of an std::vector
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t>
		t prod(const std::vector<t> &arr)
		{
			uint64 res = 1;
			for (const auto &val : arr)
				res *= val;
			return res;
		}
	}
}
