#pragma once

#include "../internal.h"

namespace rapid
{
	namespace ndarray
	{
		namespace imp
		{
			template<typename t>
			inline t rapid_dot(uint64_t len,
							   const t *__restrict a,
							   const t *__restrict b)
			{
				static_assert(false, "Invalid type for vectorDot");
			}

			template<>
			inline double rapid_dot(uint64_t len,
									const double *__restrict a,
									const double *__restrict b)
			{
				return cblas_ddot((blasint) len, a, (blasint) 1, b, (blasint) 1);
			}

			template<>
			inline float rapid_dot(uint64_t len,
								   const float *__restrict a,
								   const float *__restrict b)
			{
				return cblas_sdot((blasint) len, a, (blasint) 1, b, (blasint) 1);
			}

			template<typename t>
			inline void rapid_gemm(uint64_t M, uint64_t N, uint64_t K,
								   const t *__restrict a,
								   const t *__restrict b,
								   t *__restrict c)
			{
				static_assert(false, "Invalid type for vectorDot");
			}

			template<>
			inline void rapid_gemm(uint64_t M, uint64_t N, uint64_t K,
								   const double *__restrict a,
								   const double *__restrict b,
								   double *__restrict c)
			{
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (blasint) M, (blasint) K, (blasint) N,
							1., a, (blasint) N, b, (blasint) K, 0., c, (blasint) K);
			}

			template<>
			inline void rapid_gemm(uint64_t M, uint64_t N, uint64_t K,
								   const float *__restrict a,
								   const float *__restrict b,
								   float *__restrict c)
			{
				cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (blasint) M, (blasint) K, (blasint) N, 1., a, (blasint) N, b, (blasint) K, 0., c, (blasint) K);
			}
		}
	}
}
