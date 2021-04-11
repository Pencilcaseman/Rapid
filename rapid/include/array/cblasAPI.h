#pragma once

#include "../internal.h"

namespace rapid
{
	namespace ndarray
	{
		namespace imp
		{
			template<typename t>
			inline t rapid_dot(uint64 len,
							   const t *__restrict a,
							   const t *__restrict b)
			{
				static_assert(false, "Invalid type for vectorDot");
			}

			template<>
			inline float64 rapid_dot(uint64 len,
									const float64 *__restrict a,
									const float64 *__restrict b)
			{
				return cblas_ddot((blasint) len, a, (blasint) 1, b, (blasint) 1);
			}

			template<>
			inline float32 rapid_dot(uint64 len,
								   const float32 *__restrict a,
								   const float32 *__restrict b)
			{
				return cblas_sdot((blasint) len, a, (blasint) 1, b, (blasint) 1);
			}

			template<typename t>
			inline void rapid_gemm(uint64 M, uint64 N, uint64 K,
								   const t *__restrict a,
								   const t *__restrict b,
								   t *__restrict c)
			{
				static_assert(false, "Invalid type for vectorDot");
			}

			template<>
			inline void rapid_gemm(uint64 M, uint64 N, uint64 K,
								   const float64 *__restrict a,
								   const float64 *__restrict b,
								   float64 *__restrict c)
			{
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (blasint) M, (blasint) K, (blasint) N,
							1., a, (blasint) N, b, (blasint) K, 0., c, (blasint) K);
			}

			template<>
			inline void rapid_gemm(uint64 M, uint64 N, uint64 K,
								   const float32 *__restrict a,
								   const float32 *__restrict b,
								   float32 *__restrict c)
			{
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (blasint) M, (blasint) K, (blasint) N, 1., a, (blasint) N, b, (blasint) K, 0., c, (blasint) K);
			}
		}
	}
}
