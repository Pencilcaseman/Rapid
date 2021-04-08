#pragma once

#include "../internal.h"

namespace rapid
{
	namespace matrix
	{
		class Matrix4x4
		{
		public:
			double p00, p01, p02, p03;
			double p10, p11, p12, p13;
			double p20, p21, p22, p23;
			double p30, p31, p32, p33;

			Matrix4x4()
			{
				p00 = 0;
				p01 = 0;
				p02 = 0;
				p03 = 0;
				p10 = 0;
				p11 = 0;
				p12 = 0;
				p13 = 0;
				p20 = 0;
				p21 = 0;
				p22 = 0;
				p23 = 0;
				p30 = 0;
				p31 = 0;
				p32 = 0;
				p33 = 0;
			}

			explicit Matrix4x4(const double &val)
			{
				p00 = val;
				p01 = val;
				p02 = val;
				p03 = val;

				p10 = val;
				p11 = val;
				p12 = val;
				p13 = val;

				p20 = val;
				p21 = val;
				p22 = val;
				p23 = val;

				p30 = val;
				p31 = val;
				p32 = val;
				p33 = val;
			}

			Matrix4x4(const double &_p00,
					  const double &_p01,
					  const double &_p02,
					  const double &_p03,
					  const double &_p10,
					  const double &_p11,
					  const double &_p12,
					  const double &_p13,
					  const double &_p20,
					  const double &_p21,
					  const double &_p22,
					  const double &_p23,
					  const double &_p30,
					  const double &_p31,
					  const double &_p32,
					  const double &_p33)
			{
				p00 = _p00;
				p01 = _p01;
				p02 = _p02;
				p03 = _p03;

				p10 = _p10;
				p11 = _p11;
				p12 = _p12;
				p13 = _p13;

				p20 = _p20;
				p21 = _p21;
				p22 = _p22;
				p23 = _p23;

				p30 = _p30;
				p31 = _p31;
				p32 = _p32;
				p33 = _p33;
			}

			static Matrix4x4 identity()
			{
				return {1, 0, 0, 0,
						0, 1, 0, 0,
						0, 0, 1, 0,
						0, 0, 0, 1};
			}

			inline Matrix4x4 transposed() const
			{
				return {p00, p10, p20, p30,
						p01, p11, p21, p31,
						p02, p12, p22, p32,
						p03, p13, p23, p33};
			}

			inline Matrix4x4 operator+(const Matrix4x4 &other) const
			{
				return {p00 + other.p00, p01 + other.p01, p02 + other.p02, p03 + other.p03,
						p10 + other.p10, p11 + other.p11, p12 + other.p12, p13 + other.p13,
						p20 + other.p20, p21 + other.p21, p22 + other.p22, p23 + other.p23,
						p30 + other.p30, p31 + other.p31, p32 + other.p32, p33 + other.p33};
			}

			inline Matrix4x4 operator-(const Matrix4x4 &other) const
			{
				return {p00 - other.p00, p01 - other.p01, p02 - other.p02, p03 - other.p03,
						p10 - other.p10, p11 - other.p11, p12 - other.p12, p13 - other.p13,
						p20 - other.p20, p21 - other.p21, p22 - other.p22, p23 - other.p23,
						p30 - other.p30, p31 - other.p31, p32 - other.p32, p33 - other.p33};
			}

			inline Matrix4x4 operator*(const Matrix4x4 &other) const
			{
				return {p00 * other.p00, p01 * other.p01, p02 * other.p02, p03 * other.p03,
						p10 * other.p10, p11 * other.p11, p12 * other.p12, p13 * other.p13,
						p20 * other.p20, p21 * other.p21, p22 * other.p22, p23 * other.p23,
						p30 * other.p30, p31 * other.p31, p32 * other.p32, p33 * other.p33};
			}

			inline Matrix4x4 operator/(const Matrix4x4 &other) const
			{
				return {p00 / other.p00, p01 / other.p01, p02 / other.p02, p03 / other.p03,
						p10 / other.p10, p11 / other.p11, p12 / other.p12, p13 / other.p13,
						p20 / other.p20, p21 / other.p21, p22 / other.p22, p23 / other.p23,
						p30 / other.p30, p31 / other.p31, p32 / other.p32, p33 / other.p33};
			}

			inline Matrix4x4 dot(const Matrix4x4 &d) const
			{
				return {p00 * d.p00 + p01 * d.p10 + p02 * d.p20 + p03 * d.p30,
						p00 * d.p01 + p01 * d.p11 + p02 * d.p21 + p03 * d.p31,
						p00 * d.p02 + p01 * d.p12 + p02 * d.p22 + p03 * d.p32,
						p00 * d.p03 + p01 * d.p13 + p02 * d.p23 + p03 * d.p33,

						p10 * d.p00 + p11 * d.p10 + p12 * d.p20 + p13 * d.p30,
						p10 * d.p01 + p11 * d.p11 + p12 * d.p21 + p13 * d.p31,
						p10 * d.p02 + p11 * d.p12 + p12 * d.p22 + p13 * d.p32,
						p10 * d.p03 + p11 * d.p13 + p12 * d.p23 + p13 * d.p33,

						p20 * d.p00 + p21 * d.p10 + p22 * d.p20 + p23 * d.p30,
						p20 * d.p01 + p21 * d.p11 + p22 * d.p21 + p23 * d.p31,
						p20 * d.p02 + p21 * d.p12 + p22 * d.p22 + p23 * d.p32,
						p20 * d.p03 + p21 * d.p13 + p22 * d.p23 + p23 * d.p33,

						p30 * d.p00 + p31 * d.p10 + p32 * d.p20 + p33 * d.p30,
						p30 * d.p01 + p31 * d.p11 + p32 * d.p21 + p33 * d.p31,
						p30 * d.p02 + p31 * d.p12 + p32 * d.p22 + p33 * d.p32,
						p30 * d.p03 + p31 * d.p13 + p32 * d.p23 + p33 * d.p33
				};
			}

			inline std::string toString() const
			{
				return "[[" + std::to_string(p00) + ", " + std::to_string(p01) + ", " + std::to_string(p02) + ", " + std::to_string(p03) + "]\n" +
					" [" + std::to_string(p10) + ", " + std::to_string(p11) + ", " + std::to_string(p12) + ", " + std::to_string(p13) + "]\n" +
					" [" + std::to_string(p20) + ", " + std::to_string(p21) + ", " + std::to_string(p22) + ", " + std::to_string(p23) + "]\n" +
					" [" + std::to_string(p30) + ", " + std::to_string(p31) + ", " + std::to_string(p32) + ", " + std::to_string(p33) + "]]\n";
			}
		};
	}
}