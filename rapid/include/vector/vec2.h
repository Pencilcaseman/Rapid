#pragma once

#include "../internal.h"

namespace rapid
{
	namespace vector
	{
		template<typename vecType>
		class Vec2
		{
		public:
			vecType x{};
			vecType y{};

			Vec2() = default;

			Vec2(const vecType xx, const vecType yy) : x(xx), y(yy)
			{}

			inline Vec2 operator+(const Vec2 &other) const
			{
				return {x + other.x, y + other.y};
			}

			inline Vec2 operator-(const Vec2 &other) const
			{
				return {x - other.x, y - other.y};
			}

			inline Vec2 operator*(const Vec2 &other) const
			{
				return {x * other.x, y * other.y};
			}

			inline Vec2 operator/(const Vec2 &other) const
			{
				return {x / other.x, y / other.y};
			}

			inline Vec2 operator+(const vecType &other) const
			{
				return {x + other, y + other};
			}

			inline Vec2 operator-(const vecType &other) const
			{
				return {x - other, y - other};
			}

			inline Vec2 operator*(const vecType &other) const
			{
				return {x * other, y * other};
			}

			inline Vec2 operator/(const vecType &other) const
			{
				return {x / other, y / other};
			}


			inline void operator+=(const Vec2 &other)
			{
				x += other.x;
				y += other.y;
			}

			inline void operator-=(const Vec2 &other)
			{
				x -= other.x;
				y -= other.y;
			}

			inline void operator*=(const Vec2 &other)
			{
				x *= other.x;
				y *= other.y;
			}

			inline void operator/=(const Vec2 &other)
			{
				x /= other.x;
				y /= other.y;
			}

			inline void operator+=(const vecType &other)
			{
				x += other;
				y += other;
			}

			inline void operator-=(const vecType &other)
			{
				x -= other;
				y -= other;
			}

			inline void operator*=(const vecType &other)
			{
				x *= other;
				y *= other;
			}

			inline void operator/=(const vecType &other)
			{
				x /= other;
				y /= other;
			}

			inline vecType mag() const
			{
				return sqrt(x * x + y * y);
			}

			inline vecType magSquared() const
			{
				return x * x + y * y;
			}

			inline Vec2 yx() const
			{
				return {y, x};
			}

			std::string toString() const
			{
				return "Vec2(X: " + std::to_string(x) + ", Y: " + std::to_string(y) + ")";
			}
		};
	}
}
