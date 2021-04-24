#pragma once

#include "../internal.h"
#include "../rapid_math.h"
#include "../io.h"

#include "fromData.h"

#ifndef RAPID_NO_BLAS
#include "cblasAPI.h"
#endif

#ifdef RAPID_CUDA
#include "cudaAPI.cuh"
#endif

namespace rapid
{
	namespace ndarray
	{
	#ifdef RAPID_CUDA
		namespace handle
		{
			static cublasHandle_t handle;
			bool handleInitialized = false;

			inline void createHandle()
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
				handleInitialized = true;
			}
		}
	#endif

		namespace utils
		{
			/// <summary>
			/// Convert an index and a given array shape and return the memory
			/// offset for the given position in contiguous memory
			/// </summary>
			/// <typeparam name="indexT"></typeparam>
			/// <typeparam name="shapeT"></typeparam>
			/// <param name="index"></param>
			/// <param name="shape"></param>
			/// <returns></returns>
			template<typename indexT, typename shapeT>
			inline indexT ndToScalar(const std::vector<indexT> &index,
									 const std::vector<shapeT> &shape)
			{
				indexT sig = 1;
				indexT pos = 0;

				for (indexT i = shape.size(); i > 0; i--)
				{
					pos += (i - 1 < index.size() ? index[i - 1] : 0) * sig;
					sig *= shape[i - 1];
				}

				return pos;
			}

			/// <summary>
			/// Convert an index and a given array shape and return the memory
			/// offset for the given position in contiguous memory
			/// </summary>
			/// <typeparam name="indexT"></typeparam>
			/// <typeparam name="shapeT"></typeparam>
			/// <param name="index"></param>
			/// <param name="shape"></param>
			/// <returns></returns>
			template<typename indexT, typename shapeT>
			inline indexT ndToScalar(const std::initializer_list<indexT> &index,
									 const std::vector<shapeT> &shape)
			{
				indexT sig = 1;
				indexT pos = 0;
				uint64 off;

				for (indexT i = shape.size(); i > 0; i--)
				{
					off = i - 1;
					pos += (i - 1 < index.size() ? (*(index.begin() + off)) : 0) * sig;
					sig *= shape[off];
				}

				return pos;
			}

			inline std::vector<uint64> transposedShape(const std::vector<uint64> &shape, const std::vector<uint64> &order)
			{
				std::vector<uint64> newDims;

				newDims = std::vector<uint64>(shape.size());
				if (order.empty())
					for (uint64 i = 0; i < shape.size(); i++)
						newDims[i] = shape[shape.size() - i - 1];
				else
					for (uint64 i = 0; i < shape.size(); i++)
						newDims[i] = shape[order[i]];

				return newDims;
			}

			template<typename _Ty>
			inline std::vector<_Ty> subVector(const std::vector<_Ty> &vec, uint64 start = (uint64) -1, uint64 end = (uint64) -1)
			{
				auto s = vec.begin();
				auto e = vec.end();

				if (start != (uint64) -1) s += start;
				if (end != (uint64) -1) e -= end;

				return std::vector<_Ty>(s, e);
			}
		}

		namespace imp
		{
			/// <summary>
			/// Convert a set of dimensions into a memory location. Intended for internal use only
			/// </summary>
			/// <param name="dims"></param>
			/// <param name="pos"></param>
			/// <returns></returns>
			uint64 dimsToIndex(const std::vector<uint64> &dims, const std::vector<uint64> &pos)
			{
				uint64 index = 0;
				for (long int i = 0; i < dims.size(); i++)
				{
					uint64 sub = pos[i];
					for (uint64 j = i; j < dims.size() - 1; j++)
						sub *= dims[j + 1];
					index += sub;
				}
				return index;
			}
		}

		enum class ExecutionType
		{
			SERIAL = 0b0001,
			PARALLEL = 0b0010,
			MASSIVE = 0b0100
		};

	#ifdef RAPID_CUDA
		enum ArrayLocation
		{
			CPU,
			GPU
		};
	#else
		enum ArrayLocation
		{
			CPU
		};
	#endif

		/// <summary>
		/// A powerful and fast ndarray type, supporting a wide variety
		/// of optimized functions and routines. It also supports different
		/// arrayTypes, allowing for greater flexibility.
		/// </summary>
		/// <typeparam name="arrayType"></typeparam>
		template<typename arrayType, ArrayLocation location = CPU>
		class Array
		{
		public:
			std::vector<uint64> shape;
			arrayType *dataOrigin = nullptr;
			arrayType *dataStart = nullptr;
			uint64 *originCount = nullptr;
			bool isZeroDim;

			// #ifdef RAPID_CUDA
			// 	bool useMatrixData = false;
			// 	uint64 matrixRows = 0;
			// 	uint64 matrixAccess = 0;
			// #endif

				/// <summary>
				/// Apply a lambda function to two arrays, storing the result in a third.
				/// Both arrays must be the same size, but this in not checked when running,
				/// so it is therefore the responsibility of the user to ensure this function
				/// is called safely
				/// </summary>
				/// <typeparam name="Lambda"></typeparam>
				/// <param name="a"></param>
				/// <param name="b"></param>
				/// <param name="c"></param>
				/// <param name="mode"></param>
				/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void binaryOpArrayArray(const Array<arrayType, loc> &a, const Array<arrayType, loc> &b,
												  Array<arrayType, loc> &c, ExecutionType mode, Lambda func)
			{
				uint64 size = math::prod(a.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					uint64 index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							c.dataStart[index + 0] = func(a.dataStart[index + 0], b.dataStart[index + 0]);
							c.dataStart[index + 1] = func(a.dataStart[index + 1], b.dataStart[index + 1]);
							c.dataStart[index + 2] = func(a.dataStart[index + 2], b.dataStart[index + 2]);
							c.dataStart[index + 3] = func(a.dataStart[index + 3], b.dataStart[index + 3]);
						}
					}

					for (; index < size; index++)
						c.dataStart[index] = func(a.dataStart[index], b.dataStart[index]);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b, c, func) private(index) default(none)
					for (index = 0; index < size; ++index)
						c.dataStart[index] = func(a.dataStart[index], b.dataStart[index]);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Apply a lambda function to an array in the format
			/// func(array, scalar) and store the result. Both arrays
			/// must be the same size, but this in not checked when running,
			/// so it is therefore the responsibility of the user to ensure
			/// this function is called safely
			/// </summary>
			/// <typeparam name="Lambda"></typeparam>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="mode"></param>
			/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void binaryOpArrayScalar(const Array<arrayType, loc> &a, const arrayType &b,
												   Array<arrayType, loc> &c, ExecutionType mode, Lambda func)
			{
				uint64 size = math::prod(a.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					uint64 index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							c.dataStart[index + 0] = func(a.dataStart[index + 0], b);
							c.dataStart[index + 1] = func(a.dataStart[index + 1], b);
							c.dataStart[index + 2] = func(a.dataStart[index + 2], b);
							c.dataStart[index + 3] = func(a.dataStart[index + 3], b);
						}
					}

					for (; index < size; index++)
						c.dataStart[index] = func(a.dataStart[index], b);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b, c, func) private(index) default(none)
					for (index = 0; index < size; ++index)
						c.dataStart[index] = func(a.dataStart[index], b);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Apply a lambda function to a scalar and an array in the format
			/// func(scalar, array) and store the result. Both arrays
			/// must be the same size, but this in not checked when running,
			/// so it is therefore the responsibility of the user to ensure
			/// this function is called safely
			/// </summary>
			/// <typeparam name="Lambda"></typeparam>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="c"></param>
			/// <param name="mode"></param>
			/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void binaryOpScalarArray(const arrayType &a, const Array<arrayType, loc> &b,
												   Array<arrayType, loc> &c, ExecutionType mode, Lambda func)
			{
				uint64 size = math::prod(b.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					uint64 index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							c.dataStart[index + 0] = func(a, b.dataStart[index + 0]);
							c.dataStart[index + 1] = func(a, b.dataStart[index + 1]);
							c.dataStart[index + 2] = func(a, b.dataStart[index + 2]);
							c.dataStart[index + 3] = func(a, b.dataStart[index + 3]);
						}
					}

					for (; index < size; index++)
						c.dataStart[index] = func(a, b.dataStart[index]);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b, c, func) private(index) default(none)
					for (index = 0; index < size; ++index)
						c.dataStart[index] = func(a, b.dataStart[index]);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Apply a lambda function to an array in the format
			/// func(array) and store the result
			/// </summary>
			/// <typeparam name="Lambda"></typeparam>
			/// <param name="a"></param>
			/// <param name="b"></param>
			/// <param name="mode"></param>
			/// <param name="func"></param>
			template<typename Lambda, ArrayLocation loc>
			inline static void unaryOpArray(const Array<arrayType, loc> &a, Array<arrayType, loc> &b,
											ExecutionType mode, Lambda func)
			{
				uint64 size = math::prod(a.shape);

				if (mode == ExecutionType::SERIAL)
				{
					// Serial execution on CPU
					uint64 index = 0;

					if (size > 3)
					{
						for (index = 0; index < size - 3; index += 4)
						{
							b.dataStart[index + 0] = func(a.dataStart[index + 0]);
							b.dataStart[index + 1] = func(a.dataStart[index + 1]);
							b.dataStart[index + 2] = func(a.dataStart[index + 2]);
							b.dataStart[index + 3] = func(a.dataStart[index + 3]);
						}
					}

					for (; index < size; index++)
						b.dataStart[index] = func(a.dataStart[index]);
				}
				else if (mode == ExecutionType::PARALLEL)
				{
					// Parallel execution on CPU
					long index = 0;

				#pragma omp parallel for shared(size, a, b, func) private(index) default(none)
					for (index = 0; index < size; ++index)
						b.dataStart[index] = func(a.dataStart[index]);
				}
				else
				{
					message::RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
				}
			}

			/// <summary>
			/// Resize an array to different dimensions and return the result.
			/// The data stored in the array is copied, so an update in the
			/// result array will not trigger an update in the parent.
			/// </summary>
			/// <param name="newShape"></param>
			/// <returns></returns>
			inline Array<arrayType, location> internal_resized(const std::vector<uint64> &newShape) const
			{
				rapidAssert(newShape.size() == 2, "Resizing currently only supports 2D array");

				Array<arrayType, location> res(newShape);
				auto resData = res.dataStart;
				auto thisData = dataStart;

				if (location == CPU)
				{
					for (uint64 i = 0; i < rapid::math::min(shape[0], newShape[0]); i++)
						memcpy(resData + i * newShape[1], thisData + i * shape[1],
							   sizeof(arrayType) * rapid::math::min(shape[1], newShape[1]));
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					for (uint64 i = 0; i < rapid::math::min(shape[0], newShape[0]); i++)
						cudaSafeCall(cudaMemcpy(resData + i * newShape[1], thisData + i * shape[1],
									 sizeof(arrayType) * rapid::math::min(shape[1], newShape[1]), cudaMemcpyHostToDevice));
				}
			#endif

				return res;
			}

			/// <summary>
			/// Resize an array to different dimensions and return the result.
			/// The data stored in the array is copied, so an update in the
			/// result array will not trigger an update in the parent.
			/// </summary>
			/// <param name="newShape"></param>
			/// <returns></returns>
			inline void internal_resize(const std::vector<uint64> &newShape)
			{
				auto newThis = internal_resized(newShape);

				freeSelf();

				originCount = newThis.originCount;
				(*originCount)++;

				dataOrigin = newThis.dataOrigin;
				dataStart = newThis.dataStart;

				shape = newShape;
			}

			static int calculateArithmeticMode(const std::vector<uint64> &a, const std::vector<uint64> &b)
			{
				// Check for direct or indirect shape match
				int mode = -1; // Addition mode

				uint64 aSize = a.size();
				uint64 bSize = b.size();

				uint64 prodA = math::prod(a);
				uint64 prodB = math::prod(b);

				if (a == b)
				{
					// Check for exact shape match
					mode = 0;
				}
				else if (aSize < bSize &&
						 prodA == prodB &&
						 a == utils::subVector(b, bSize - aSize))
				{
					// Check if last dimensions of other match *this, and all other dimensions are 1
					// E.g. [1 2] + [[[3 4]]] => [4 6]
					mode = 0;
				}
				else if (aSize > bSize &&
						 prodA == prodB &&
						 utils::subVector(a, aSize - bSize) == b)
				{
					// Check if last dimensions of *this match other, and all other dimensions are 1
					// E.g. [[[1 2]]] + [3 4] => [[[4 6]]]
					mode = 0;
				}
				else if (prodB == 1)
				{
					// Check if other is a single value array
					// E.g. [1 2 3] + [10] => [11 12 13]

					mode = 1;
				}
				else if (prodA == 1)
				{
					// Check if this is a single value array
					// E.g. [10] + [1 2 3] => [11 12 13]

					mode = 2;
				}
				else if (utils::subVector(a, 1) == b)
				{
					// Check for "row by row" addition
					// E.g. [[1 2]   +   [5 6]    =>   [[ 6  8]
					//       [3 4]]                     [ 8 10]]
					mode = 3;
				}
				else if (a == utils::subVector(b, 1))
				{
					// Check for reverse "row by row" addition
					// E.g. [1 2]  +   [[3 4]     =>   [[4 6]
					//                  [5 6]]          [6 8]]
					mode = 4;
				}
				else if (prodA == prodB &&
						 prodA == a[0] &&
						 a[0] == b[bSize - 1])
				{
					// Check for grid addition
					// E.g. [[1]    +    [3 4]    =>    [[4 5]
					//       [2]]                        [5 6]]
					mode = 5;
				}
				else if (prodA == prodB &&
						 prodB == b[0] &&
						 a[aSize - 1] == b[0])
				{
					// Check for reverse grid addition
					// E.g. [1 2]   +    [[3]     =>    [[4 5]
					//                    [4]]           [5 6]]
					mode = 6;
				}
				else if (a[aSize - 1] == 1 && utils::subVector(a, 0, aSize - 1) == utils::subVector(b, 0, bSize - 1))
				{
					// Check for "column by column" addition
					// E.g. [[1]     +    [[10 11]      =>     [[11 12]
					//       [2]]          [12 13]]             [14 15]]
					mode = 7;
				}
				else if (b[bSize - 1] == 1 && utils::subVector(a, 0, aSize - 1) == utils::subVector(b, 0, bSize - 1))
				{
					// Check for reverse "column by column" addition
					// E.g.  [[1 2]    +    [[5]      =>     [[ 6  7]
					//        [3 4]]         [6]]             [ 9 10]]
					mode = 8;
				}

				return mode;
			}

		public:

			/// <summary>
			/// Default constructor
			/// </summary>
			Array()
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif
			}

			/// <summary>
			/// Set this array equal to another. This function exists because
			/// calling a = b results in a different function being called
			/// that gives slightly different results. The resulting array
			/// is linked to the parent array.
			/// </summary>
			/// <param name="other"></param>
			inline void set(const Array<arrayType, location> &other)
			{
				// Only delete data if originCount becomes zero
				freeSelf();

				isZeroDim = other.isZeroDim;
				shape = other.shape;

				dataStart = other.dataStart;
				dataOrigin = other.dataOrigin;

				originCount = other.originCount;
				(*originCount)++;
			}

			/// <summary>
			/// Create a new array from a given shape. This allocates entirely
			/// new data, and no existing arrays are modified in any way.
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <typeparam name="type"></typeparam>
			/// <param name="arrShape"></param>
			Array(const std::vector<uint64> &arrShape)
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif

				if (arrShape.empty() || math::prod(arrShape) == 0)
				{
					isZeroDim = true;
					shape = {1};

					if (location == CPU)
						dataStart = new arrayType[1];
				#ifdef RAPID_CUDA
					else
						cudaSafeCall(cudaMalloc(&dataStart, sizeof(arrayType)));
				#endif

					dataOrigin = dataStart;
					originCount = new uint64;
					*originCount = 1;
				}
				else
				{
					isZeroDim = false;
					shape = std::vector<uint64>(arrShape.begin(), arrShape.end());

					if (location == CPU)
						dataStart = new arrayType[math::prod(arrShape)];
				#ifdef RAPID_CUDA
					else
						cudaSafeCall(cudaMalloc(&dataStart, sizeof(arrayType) * math::prod(arrShape)));
				#endif

					dataOrigin = dataStart;
					originCount = new uint64;
					*originCount = 1;
				}
			}

			inline static Array<arrayType, location> fromScalar(const arrayType &val)
			{
				Array<arrayType, location> res;

				res.isZeroDim = true;
				res.shape = {1};

				if (location == CPU)
				{
					res.dataStart = new arrayType[1];
					res.dataStart[0] = val;
				}
			#ifdef RAPID_CUDA
				else
				{
					cudaSafeCall(cudaMalloc(&res.dataStart, sizeof(arrayType)));
					cudaSafeCall(cudaMemcpy(res.dataStart, &val, sizeof(arrayType), cudaMemcpyHostToDevice));
				}
			#endif

				res.dataOrigin = res.dataStart;
				res.originCount = new uint64;
				(*res.originCount) = 1;

				return res;
			}

			/// <summary>
			/// Create an array from an existing array. The array that is created
			/// will inherit the same data as the array it is created from, so an
			/// update in one will cause an update in the other.
			/// </summary>
			/// <param name="other"></param>
			Array(const Array<arrayType, location> &other)
			{
				isZeroDim = other.isZeroDim;
				shape = other.shape;
				dataOrigin = other.dataOrigin;
				dataStart = other.dataStart;
				originCount = other.originCount;

				if (originCount)
					(*originCount)++;
			}

			/// <summary>
			/// Set one array equal to another and copy the memory.
			/// This means an update in one array will not trigger
			/// an update in the other
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			Array<arrayType, location> &operator=(const Array<arrayType, location> &other)
			{
				if (!other.originCount)
					return *this;

				if (originCount != nullptr)
				{
					rapidAssert(shape == other.shape, "Invalid shape for array setting");

					if (location == CPU)
						memcpy(dataStart, other.dataStart, math::prod(shape) * sizeof(arrayType));
				#ifdef RAPID_CUDA
					else if (location == GPU)
					{
						cudaSafeCall(cudaMemcpy(dataStart, other.dataStart, math::prod(shape) * sizeof(arrayType), cudaMemcpyDeviceToDevice));
					}
				#endif
				}
				else
				{
					shape = std::vector<uint64>(other.shape.begin(), other.shape.end());

					if (location == CPU)
					{
						dataStart = new arrayType[math::prod(shape)];
						memcpy(dataStart, other.dataStart, math::prod(shape) * sizeof(arrayType));
					}
				#ifdef RAPID_CUDA
					else if (location == GPU)
					{
						cudaSafeCall(cudaMalloc(&dataStart, sizeof(arrayType) * math::prod(shape)));
						cudaSafeCall(cudaMemcpy(dataStart, other.dataStart, math::prod(shape) * sizeof(arrayType), cudaMemcpyDeviceToDevice));
					}
				#endif
						dataOrigin = dataStart;
						originCount = new uint64;
						(*originCount) = 1;
				}

				isZeroDim = other.isZeroDim;

				return *this;
			}

			/// <summary>
			/// Set an array equal to a scalar value. This fills
			/// the array with the value.
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			Array<arrayType, location> &operator=(const arrayType &other)
			{
				fill(other);
				isZeroDim = other.isZeroDim;
				return *this;
			}

		#ifdef RAPID_CUDA
			/// <summary>
			/// Convert the data in an array to row-major format from
			/// column-major format, and return the result. The result
			/// array is not linked in any way to the parent
			/// </summary>
			/// <returns></returns>
			inline Array<arrayType, location> toRowMajor() const
			{
				cudaSafeCall(cudaDeviceSynchronize());

				Array<arrayType, location> res(shape);

				if (shape.size() == 2)
				{
					cuda::columnToRowOrdering((unsigned int) shape[0], (unsigned int) shape[1], dataStart, res.dataStart);
				}
				else if (shape.size() > 2)
				{
					for (uint64 i = 0; i < shape[0]; i++)
					{
						res[i] = (*this)[i].toRowMajor();
					}
				}
				else
				{
					cudaSafeCall(cudaMemcpy(res.dataStart, dataStart, sizeof(arrayType) * math::prod(shape), cudaMemcpyDeviceToDevice));
				}

				return res;
			}

			/// <summary>
			/// Column to row-major ordering in place
			/// </summary>
			inline void toRowMajor_inplace() const
			{
				cudaSafeCall(cudaDeviceSynchronize());

				if (shape.size() == 2)
				{
					cuda::columnToRowOrdering((unsigned int) shape[0], (unsigned int) shape[1], dataStart, dataStart);
				}
				else if (shape.size() > 2)
				{
					for (uint64 i = 0; i < shape[0]; i++)
					{
						(*this)[i].toRowMajor_inplace();
					}
				}
			}

			/// <summary>
			/// Row to column-major ordering in place
			/// </summary>
			inline void toColumMajor_inplace() const
			{
				cudaSafeCall(cudaDeviceSynchronize());

				if (shape.size() == 2)
				{
					cuda::rowToColumnOrdering((unsigned int) shape[0], (unsigned int) shape[1], dataStart, dataStart);
				}
				else if (shape.size() > 2)
				{
					for (uint64 i = 0; i < shape[0]; i++)
					{
						(*this)[i].toColumMajor_inplace();
					}
				}
			}
		#endif

			/// <summary>
			/// Create an array from the provided data, without creating a
			/// temporary one first. This fixes memory leaks and is intended
			/// for internal use only.
			/// </summary>
			/// <param name="arrDims"></param>
			/// <param name="newDataOrigin"></param>
			/// <param name="dataStart"></param>
			/// <param name="originCount"></param>
			/// <param name="isZeroDim"></param>
			/// <returns></returns>
			static inline Array<arrayType, location> fromData(const std::vector<uint64> &arrDims,
															  arrayType *newDataOrigin, arrayType *dataStart,
															  uint64 *originCount, bool isZeroDim)
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif

				Array<arrayType, location> res;
				res.isZeroDim = isZeroDim;
				res.shape = std::vector<uint64>(arrDims.begin(), arrDims.end());
				res.dataOrigin = newDataOrigin;
				res.dataStart = dataStart;
				res.originCount = originCount;
				return res;
			}

			/// <summary>
			/// Create a new array from an initializer_list. This supports creating
			/// arrays of up to 20-dimensions via nested initializer lists
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="data"></param>
			/// <returns></returns>
			template<typename t>
			static inline Array<arrayType, location> fromData(const std::initializer_list<t> &data)
			{
			#ifdef RAPID_CUDA
				if (!handle::handleInitialized)
					handle::createHandle();
			#endif

				auto res = Array<arrayType, location>({data.size()});
				std::vector<arrayType> values;

				for (const auto &val : data)
					values.emplace_back(val);

				if (location == CPU)
					memcpy(res.dataStart, values.data(), sizeof(arrayType) * values.size());
			#ifdef RAPID_CUDA
				else if (location == GPU)
					cudaSafeCall(cudaMemcpy(res.dataStart, values.data(), sizeof(arrayType) * values.size(), cudaMemcpyHostToDevice));
			#endif

				return res;
			}

		#define imp_temp template<typename t>
		#define imp_func_def(x) static inline Array<arrayType, location> fromData(const x &data)
		#define imp_func_body	auto res = Array<arrayType, location>(imp::extractShape(data)); \
							    uint64 index = 0; \
								for (const auto &val : data) res[index++] = Array<arrayType, location>::fromData(val); \
									return res;
		#define L std::initializer_list

			// Up to 20-dimensional array setting from data
			imp_temp imp_func_def(L<L<t>>)
			{
				auto res = Array<arrayType, location>(imp::extractShape(data));

				uint64 index = 0;
				for (const auto &val : data)
					res[index++] = Array<arrayType, location>::fromData(val);

				return res;
			}

			imp_temp imp_func_def(L<L<L<t>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<t>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<t>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<t>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<t>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<t>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<t>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}
			imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>>)
			{
				imp_func_body
			}

		#undef imp_temp
		#undef imp_func_def
		#undef imp_func_body
		#undef L

			~Array()
			{
				freeSelf();
			}

			/// <summary>
			/// Free the contents of the array
			/// </summary>
			inline void freeSelf()
			{
				// Ensure the array is initialized
				if (originCount)
				{
					// Only delete data if originCount becomes zero
					(*originCount)--;

					if ((*originCount) == 0)
					{
						if (location == CPU)
							delete[] dataOrigin;
					#ifdef RAPID_CUDA
						else
							cudaSafeCall(cudaFree(dataOrigin));
					#endif
						delete originCount;
					}
				}
			}

			/// <summary>
			/// Cast a zero-dimensional array to a scalar value
			/// </summary>
			/// <typeparam name="t"></typeparam>
			template<typename t> //, typename std::enable_if<std::is_floating_point<t>::value || std::is_integral<t>::value>::type = 0>
			inline operator t() const
			{
			#ifdef RAPID_DEBUG
				if (!isZeroDim)
					rapidAssert(isZeroDim, "Cannot cast multidimensional array to scalar value");
			#endif
			
				if (location == CPU)
					return (t) (dataStart[0]);
			
			#ifdef RAPID_CUDA
				if (location == GPU)
				{
					cudaSafeCall(cudaDeviceSynchronize());
					arrayType res;
					cudaSafeCall(cudaMemcpy(&res, dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));
					return res;
				}
			#endif
			}

			inline bool isInitialized() const
			{
				return originCount != nullptr;
			}

			/// <summary>
			/// Access a subarray or value of an array. The result is linked
			/// to the parent array, so an update in one will trigger an update
			/// in the other.
			/// </summary>
			/// <param name="index"></param>
			/// <returns></returns>
			Array<arrayType, location> operator[](const uint64 &index) const
			{
				rapidAssert(index < shape[0], "Index out of range for array subscript");

				(*originCount)++;

				if (shape.size() == 1)
				{
					return Array<arrayType, location>::fromData({1}, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, true);
				}

				if (location == CPU)
				{
					std::vector<uint64> resShape(shape.begin() + 1, shape.end());
					return Array<arrayType, location>::fromData(resShape, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, isZeroDim);
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					std::vector<uint64> resShape(shape.begin() + 1, shape.end());
					return Array<arrayType, location>::fromData(resShape, dataOrigin, dataStart + utils::ndToScalar({index}, shape),
																originCount, isZeroDim);
				}
			#endif
			}

			/// <summary>
			/// Directly access an individual value in an array. This does
			/// not allow for changing the value, but is much faster than
			/// accessing it via repeated subscript operations
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="index"></param>
			/// <returns></returns>
			template<typename t>
			inline arrayType accessVal(const std::initializer_list<t> &index) const
			{
				rapidAssert(index.size() == shape.size(), "Invalid number of dimensions to access");
			#ifdef RAPID_DEBUG
				for (uint64 i = 0; i < index.size(); i++)
				{
					if (*(index.begin() + i) < 0 || *(index.begin() + i) >= shape[i])
						message::RapidError("Index Error", "Index out of range or negative").display();
				}
			#endif

				if (location == CPU)
					return dataStart[utils::ndToScalar(index, shape)];
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					arrayType res;
					cudaSafeCall(cudaDeviceSynchronize());
					cudaSafeCall(cudaMemcpy(&res, dataStart + utils::ndToScalar(index, shape), sizeof(arrayType), cudaMemcpyDeviceToHost));
					return res;
				}
			#endif
			}

			/// <summary>
			/// Set a scalar value in an array from a given
			/// index location
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="index"></param>
			/// <param name="val"></param>
			template<typename t>
			inline void setVal(const std::initializer_list<t> &index, const arrayType &val) const
			{
				rapidAssert(index.size() == shape.size(), "Invalid number of dimensions to access");
			#ifdef RAPID_DEBUG
				for (uint64 i = 0; i < index.size(); i++)
				{
					if (*(index.begin() + i) < 0 || *(index.begin() + i) >= shape[i])
						message::RapidError("Index Error", "Index out of range or negative");
				}
			#endif

				if (location == CPU)
					dataStart[utils::ndToScalar(index, shape)] = val;
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cudaSafeCall(cudaDeviceSynchronize());
					cudaSafeCall(cudaMemcpy(dataStart + utils::ndToScalar(index, shape), &val, sizeof(arrayType), cudaMemcpyHostToDevice));
				}
			#endif
			}

			inline Array<arrayType, location> operator-() const
			{
				auto res = Array<arrayType, location>(shape);

				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);

					Array<arrayType, location>::unaryOpArray(*this, res,
															 math::prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
															 [](arrayType x)
					{
						return -x;
					});

					return res;
				}

			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					auto res = Array<arrayType, location>(shape);
					cuda::sub_scalar_array((unsigned int) math::prod(shape), 0, dataStart, 1, res.dataStart, 1);
					return res;
				}
			#endif
			}

			/// <summary>
			/// Array add Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator+(const Array<arrayType, location> &other) const
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(mode != -1, "Cannot add arrays with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x + y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);

								cuda::add_array_array((unsigned int) math::prod(shape), dataStart, 1, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x + y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::add_array_scalar((unsigned int) math::prod(shape), dataStart, 1, val, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 2:
						{
							// Cases:
							//  > This is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(other.shape);

								Array<arrayType, location>::binaryOpScalarArray(dataStart[0], other, res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x + y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(other.shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));
								cuda::add_scalar_array((unsigned int) math::prod(other.shape), val, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" addition

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < shape[0]; i++)
								res[i] = (*this)[i] + other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 4:
						{
							// Cases:
							//  > Reverse "row by row" addition

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < other.shape[0]; i++)
								res[i] = (*this) + other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 5:
						{
							// Cases
							//  > Grid addition

							auto resShape = std::vector<uint64>(other.shape.size() + 1);
							for (uint64 i = 0; i < other.shape.size(); i++)
								resShape[i] = shape[i];
							resShape[other.shape.size()] = other.shape[other.shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this)[i] + other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 6:
						{
							// Cases
							//  > Reverse grid addition

							auto resShape = std::vector<uint64>(shape.size() + 1);
							for (uint64 i = 0; i < shape.size(); i++)
								resShape[i] = other.shape[i];
							resShape[shape.size()] = shape[shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this) + other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 7:
						{
							// Cases
							//  > "Column by column" addition

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] + other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 8:
						{
							// Cases
							//  > "Column by column" addition

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] + other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					default:
						{
							message::RapidError("Addition Error", "Invalid addition mode '" + std::to_string(mode) + "'").display();
						}
				}
			}

			/// <summary>
			/// Array sub Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator-(const Array<arrayType, location> &other) const
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(mode != -1, "Cannot subtract arrays with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x - y;
								});

								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);

								cuda::sub_array_array((unsigned int) math::prod(shape), dataStart, 1, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x - y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::sub_array_scalar((unsigned int) math::prod(shape), dataStart, 1, val, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 2:
						{
							// Cases:
							//  > This is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(other.shape);

								Array<arrayType, location>::binaryOpScalarArray(dataStart[0], other, res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x - y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(other.shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));
								cuda::sub_scalar_array((unsigned int) math::prod(other.shape), val, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" subtraction

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < shape[0]; i++)
								res[i] = (*this)[i] - other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 4:
						{
							// Cases:
							//  > Reverse "row by row" subtraction

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < other.shape[0]; i++)
								res[i] = (*this) - other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 5:
						{
							// Cases
							//  > Grid subtraction

							auto resShape = std::vector<uint64>(other.shape.size() + 1);
							for (uint64 i = 0; i < other.shape.size(); i++)
								resShape[i] = shape[i];
							resShape[other.shape.size()] = other.shape[other.shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this)[i] - other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 6:
						{
							// Cases
							//  > Reverse grid subtraction

							auto resShape = std::vector<uint64>(shape.size() + 1);
							for (uint64 i = 0; i < shape.size(); i++)
								resShape[i] = other.shape[i];
							resShape[shape.size()] = shape[shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this) - other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 7:
						{
							// Cases
							//  > "Column by column" subtraction

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] - other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 8:
						{
							// Cases
							//  > reverse "column by column" subtraction

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] - other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					default:
						{
							message::RapidError("Subtraction Error", "Invalid subtraction mode '" + std::to_string(mode) + "'").display();
						}
				}
			}

			/// <summary>
			/// Array mul Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator*(const Array<arrayType, location> &other) const
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(mode != -1, "Cannot multiply arrays with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x * y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);

								cuda::mul_array_array((unsigned int) math::prod(shape), dataStart, 1, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x * y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::mul_array_scalar((unsigned int) math::prod(shape), dataStart, 1, val, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 2:
						{
							// Cases:
							//  > This is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(other.shape);

								Array<arrayType, location>::binaryOpScalarArray(dataStart[0], other, res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x * y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(other.shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));
								cuda::mul_scalar_array((unsigned int) math::prod(other.shape), val, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" multiplication

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < shape[0]; i++)
								res[i] = (*this)[i] * other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 4:
						{
							// Cases:
							//  > Reverse "row by row" multiplication

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < other.shape[0]; i++)
								res[i] = (*this) * other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 5:
						{
							// Cases
							//  > Grid multiplication

							auto resShape = std::vector<uint64>(other.shape.size() + 1);
							for (uint64 i = 0; i < other.shape.size(); i++)
								resShape[i] = shape[i];
							resShape[other.shape.size()] = other.shape[other.shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this)[i] * other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 6:
						{
							// Cases
							//  > Reverse grid multiplication

							auto resShape = std::vector<uint64>(shape.size() + 1);
							for (uint64 i = 0; i < shape.size(); i++)
								resShape[i] = other.shape[i];
							resShape[shape.size()] = shape[shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this) * other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 7:
						{
							// Cases
							//  > "Column by column" multiplication

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] * other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 8:
						{
							// Cases
							//  > reverse "column by column" multiplication

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] * other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					default:
						{
							message::RapidError("Multiplication Error", "Invalid multiplication mode '" + std::to_string(mode) + "'").display();
						}
				}
			}

			/// <summary>
			/// Array div Array
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> operator/(const Array<arrayType, location> &other) const
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(mode != -1, "Cannot divide arrays with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, res,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x / y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);

								cuda::div_array_array((unsigned int) math::prod(shape), dataStart, 1, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x / y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::div_array_scalar((unsigned int) math::prod(shape), dataStart, 1, val, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 2:
						{
							// Cases:
							//  > This is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(other.shape);

								Array<arrayType, location>::binaryOpScalarArray(dataStart[0], other, res,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x / y;
								});

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								auto res = Array<arrayType, location>(other.shape);
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));
								cuda::div_scalar_array((unsigned int) math::prod(other.shape), val, other.dataStart, 1, res.dataStart, 1);

								res.isZeroDim = isZeroDim && other.isZeroDim;
								return res;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" division

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < shape[0]; i++)
								res[i] = (*this)[i] / other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 4:
						{
							// Cases:
							//  > Reverse "row by row" division

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < other.shape[0]; i++)
								res[i] = (*this) / other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 5:
						{
							// Cases
							//  > Grid division

							auto resShape = std::vector<uint64>(other.shape.size() + 1);
							for (uint64 i = 0; i < other.shape.size(); i++)
								resShape[i] = shape[i];
							resShape[other.shape.size()] = other.shape[other.shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this)[i] / other;

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 6:
						{
							// Cases
							//  > Reverse grid division

							auto resShape = std::vector<uint64>(shape.size() + 1);
							for (uint64 i = 0; i < shape.size(); i++)
								resShape[i] = other.shape[i];
							resShape[shape.size()] = shape[shape.size() - 1];

							auto res = Array<arrayType, location>(resShape);

							for (uint64 i = 0; i < resShape[0]; i++)
								res[i] = (*this) / other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 7:
						{
							// Cases
							//  > "Column by column" division

							auto res = Array<arrayType, location>(other.shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] / other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					case 8:
						{
							// Cases
							//  > reverse "column by column" division

							auto res = Array<arrayType, location>(shape);

							for (uint64 i = 0; i < res.shape[0]; i++)
								res[i] = (*this)[i] / other[i];

							res.isZeroDim = isZeroDim && other.isZeroDim;
							return res;
						}
					default:
						{
							message::RapidError("Division Error", "Invalid division mode '" + std::to_string(mode) + "'").display();
						}
				}
			}

			/// <summary>
			/// Array add Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator+(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array <arrayType, location> ::binaryOpArrayScalar(*this, (arrayType) other,
																	  res, math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	  [](arrayType x, arrayType y)
					{
						return x + y;
					});

					res.isZeroDim = isZeroDim;
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);

				cuda::add_array_scalar((unsigned int) math::prod(shape), dataStart, 1, (arrayType) other, res.dataStart, 1);

				res.isZeroDim = isZeroDim;
				return res;
			#endif
			}

			/// <summary>
			/// Array sub Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator-(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayScalar(*this, (arrayType) other, res,
																	math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x - y;
					});

					res.isZeroDim = isZeroDim;
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::sub_array_scalar((unsigned int) math::prod(shape), dataStart, 1, (arrayType) other, res.dataStart, 1);

				res.isZeroDim = isZeroDim;
				return res;
			#endif
			}

			/// <summary>
			/// Array mul Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator*(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayScalar(*this, (arrayType) other, res,
																	math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x * y;
					});

					res.isZeroDim = isZeroDim;
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::mul_array_scalar(math::prod(shape), dataStart, 1, (arrayType) other, res.dataStart, 1);

				res.isZeroDim = isZeroDim;
				return res;
			#endif
			}

			/// <summary>
			/// Array div Scalar
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <param name="other"></param>
			/// <returns></returns>
			template<typename t>
			inline Array<arrayType, location> operator/(const t &other) const
			{
				if (location == CPU)
				{
					auto res = Array<arrayType, location>(shape);
					Array<arrayType, location>::binaryOpArrayScalar(*this, (arrayType) other, res,
																	math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x / y;
					});

					res.isZeroDim = isZeroDim;
					return res;
				}

			#ifdef RAPID_CUDA
				auto res = Array<arrayType, location>(shape);
				cuda::div_array_scalar(math::prod(shape), dataStart, 1, (arrayType) other, res.dataStart, 1);

				res.isZeroDim = isZeroDim;
				return res;
			#endif
			}

			inline Array<arrayType, location> &operator+=(const Array<arrayType, location> &other)
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1 ||
					mode == 2 ||
					mode == 4 ||
					mode == 5 ||
					mode == 6 ||
					mode == 7)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(false, "Cannot add arrays inplace with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x + y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								cuda::add_array_array(math::prod(shape), dataStart, 1, other.dataStart, 1, dataStart, 1);
								return *this;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], *this,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x + y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::add_array_scalar(math::prod(shape), dataStart, 1, val, dataStart, 1);

								return *this;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" addition

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] += other;

							return *this;
						}
					case 8:
						{
							// Cases
							//  > reverse "column by column" addition

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] += other[i];

							return *this;
						}
					default:
						{
							message::RapidError("Addition Error", "Invalid addition mode '" + std::to_string(mode) + "'").display();
						}
				}

				return *this;
			}

			inline Array<arrayType, location> &operator-=(const Array<arrayType, location> &other)
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1 ||
					mode == 2 ||
					mode == 4 ||
					mode == 5 ||
					mode == 6 ||
					mode == 7)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(false, "Cannot subtract arrays inplace with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x - y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								cuda::sub_array_array(math::prod(shape), dataStart, 1, other.dataStart, 1, dataStart, 1);
								return *this;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], *this,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x - y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::sub_array_scalar(math::prod(shape), dataStart, 1, val, dataStart, 1);

								return *this;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" subtraction

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] -= other;

							return *this;
						}
					case 8:
						{
							// Cases
							//  > reverse "column by column" subtraction

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] -= other[i];

							return *this;
						}
					default:
						{
							message::RapidError("Subtraction Error", "Invalid subtraction mode '" + std::to_string(mode) + "'").display();
						}
				}

				return *this;
			}

			inline Array<arrayType, location> &operator*=(const Array<arrayType, location> &other)
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1 ||
					mode == 2 ||
					mode == 4 ||
					mode == 5 ||
					mode == 6 ||
					mode == 7)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(false, "Cannot multiply arrays inplace with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x * y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								cuda::mul_array_array(math::prod(shape), dataStart, 1, other.dataStart, 1, dataStart, 1);
								return *this;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], *this,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x * y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::mul_array_scalar(math::prod(shape), dataStart, 1, val, dataStart, 1);

								return *this;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" multiplication

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] *= other;

							return *this;
						}
					case 8:
						{
							// Cases
							//  > reverse "column by column" multiplication

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] *= other[i];

							return *this;
						}
					default:
						{
							message::RapidError("Multiplication Error", "Invalid multiplication mode '" + std::to_string(mode) + "'").display();
						}
				}

				return *this;
			}

			inline Array<arrayType, location> &operator/=(const Array<arrayType, location> &other)
			{
				auto mode = calculateArithmeticMode(shape, other.shape);

			#ifdef RAPID_DEBUG
				if (mode == -1 ||
					mode == 2 ||
					mode == 4 ||
					mode == 5 ||
					mode == 6 ||
					mode == 7)
				{
					std::string shapeThis;
					std::string shapeOther;

					for (uint64 i = 0; i < shape.size(); i++)
						shapeThis += std::to_string(shape[i]) + (i == shape.size() - 1 ? "" : ", ");

					for (uint64 i = 0; i < other.shape.size(); i++)
						shapeOther += std::to_string(other.shape[i]) + (i == other.shape.size() - 1 ? "" : ", ");

					rapidAssert(false, "Cannot divide arrays inplace with shapes (" + shapeThis + ") and (" + shapeOther + ")");
				}
			#endif

				switch (mode)
				{
					case 0:
						{
							// Cases:
							//  > Exact match
							//  > End dimensions of other match this
							//  > End dimensions of this match other

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayArray(*this, other, *this,
																			   math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																			   [](arrayType x, arrayType y)
								{
									return x / y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								cuda::div_array_array(math::prod(shape), dataStart, 1, other.dataStart, 1, dataStart, 1);
								return *this;
							}
						#endif
						}
					case 1:
						{
							// Cases:
							//  > Other is a single value

							if (location == CPU)
							{
								auto res = Array<arrayType, location>(shape);

								Array<arrayType, location>::binaryOpArrayScalar(*this, other.dataStart[0], *this,
																				math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																				[](arrayType x, arrayType y)
								{
									return x / y;
								});

								return *this;
							}

						#ifdef RAPID_CUDA
							else if (location == GPU)
							{
								arrayType val;

								cudaSafeCall(cudaDeviceSynchronize());
								cudaSafeCall(cudaMemcpy(&val, other.dataStart, sizeof(arrayType), cudaMemcpyDeviceToHost));

								cuda::div_array_scalar(math::prod(shape), dataStart, 1, val, dataStart, 1);

								return *this;
							}
						#endif
						}
					case 3:
						{
							// Cases:
							//  > "Row by row" division

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] /= other;

							return *this;
						}
					case 8:
						{
							// Cases
							//  > reverse "column by column" division

							for (uint64 i = 0; i < shape[0]; i++)
								(*this)[i] /= other[i];

							return *this;
						}
					default:
						{
							message::RapidError("Division Error", "Invalid division mode '" + std::to_string(mode) + "'").display();
						}
				}

				return *this;
			}

			inline Array<arrayType, location> &operator+=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x + y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::add_array_scalar(math::prod(shape), dataStart, 1, other, dataStart, 1);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator-=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x - y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::sub_array_scalar(math::prod(shape), dataStart, 1, other, dataStart, 1);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator*=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x * y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::mul_array_scalar(math::prod(shape), dataStar, 1, other, dataStart, 1);
				}
			#endif

				return *this;
			}

			inline Array<arrayType, location> &operator/=(const arrayType &other)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::binaryOpArrayScalar(*this, other, *this,
																	math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
																	[](arrayType x, arrayType y)
					{
						return x / y;
					});
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cuda::div_array_scalar(math::prod(shape), dataStart, 1, other, dataStart, 1);
				}
			#endif

				return *this;
			}

			/// <summary>
			/// Fill an array with a scalar value
			/// </summary>
			/// <param name="val"></param>
			inline void fill(const arrayType &val)
			{
				if (location == CPU)
				{
					Array<arrayType, location>::unaryOpArray(*this, *this,
															 math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
															 [=](arrayType x)
					{
						return val;
					});
				}
			#ifdef RAPID_CUDA
				else
				{
					cuda::fill((unsigned int) math::prod(shape), dataStart, 1, val);
				}
			#endif
			}

			inline void fillRandom(const arrayType min = -1, const arrayType max = 1)
			{
				Array<arrayType, location>::unaryOpArray(*this, *this,
														 math::prod(shape) > 1000000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
														 [=](arrayType x)
				{
					return math::random<arrayType>(min, max);
				});
			}

			/// <summary>
			/// Calculate the dot product with another array. If the
			/// arrays are single-dimensional vectors, the vector math::product
			/// is used and a scalar value is returned. If the arrays are
			/// matrices, the matrix math::product is calculated. Otherwise, the
			/// dot product of the final two dimensions of the array are
			/// calculated.
			/// </summary>
			/// <param name="other"></param>
			/// <returns></returns>
			inline Array<arrayType, location> dot(const Array<arrayType, location> &other) const
			{
				// Matrix vector product
				if (utils::subVector(shape, 1) == other.shape)
				{
					std::vector<uint64> resShape;
					resShape.emplace_back(shape[0]);

					if (other.shape.size() > 1)
						resShape.insert(resShape.end(), other.shape.begin(), other.shape.end());

					auto res = Array<arrayType, location>(resShape);

					for (uint64 i = 0; i < shape[0]; i++)
						res[i] = (*this)[i].dot(other);

					return res;
				}

				// Reverse matrix vector product
				if (shape == utils::subVector(other.shape, 1))
				{
					std::vector<uint64> resShape;
					resShape.emplace_back(other.shape[0]);

					if (shape.size() > 1)
						resShape.insert(resShape.end(), shape.begin(), shape.end());

					auto res = Array<arrayType, location>(resShape);

					for (uint64 i = 0; i < shape[0]; i++)
						res[i] = other[i].dot((*this));

					return res;
				}

				rapidAssert(shape.size() == other.shape.size(), "Invalid number of dimensions for array dot product");
				uint64 dims = shape.size();

				if (location == CPU)
				{
				#ifndef RAPID_NO_BLAS
					switch (dims)
					{
						case 1:
							{
								rapidAssert(shape[0] == other.shape[0], "Invalid shape for array math::product");
								rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array math::product");

								Array<arrayType, location> res(shape);
								res.isZeroDim = true;
								res.dataStart[0] = imp::rapid_dot(shape[0], dataStart, other.dataStart);

								return res;
							}
						case 2:
							{
								rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot math::product");

								Array<arrayType, location> res({shape[0], other.shape[1]});

								const uint64 M = shape[0];
								const uint64 N = shape[1];
								const uint64 K = other.shape[1];

								const arrayType *a = dataStart;
								const arrayType *b = other.dataStart;
								arrayType *c = res.dataStart;

								imp::rapid_gemm(M, N, K, a, b, c);

								return res;
							}
						default:
							{
								std::vector<uint64> resShape = shape;
								resShape[resShape.size() - 2] = shape[shape.size() - 2];
								resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
								Array<arrayType, location> res(resShape);

								for (uint64 i = 0; i < shape[0]; i++)
								{
									res[i] = (*this)[i].dot(other[i]);
								}

								return res;
							}
					}
				#else
				#ifndef RAPID_NO_AMP
					using namespace concurrency;
				#endif

					switch (dims)
					{
						case 1:
							{
								rapidAssert(shape[0] == other.shape[0], "Invalid shape for array math::product");
								rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array math::product");

								Array<arrayType, location> res({1});
								res.isZeroDim = true;
								res.dataStart[0] = 0;

								for (uint64 i = 0; i < shape[0]; i++)
									res.dataStart[0] += dataStart[i] * other.dataStart[i];

								return res;
							}
						case 2:
							{
								rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot math::product");
								uint64 mode;
								uint64 size = shape[0] * shape[1] * other.shape[1];

								if (size < 8000) mode = 0;
								else if (size < 64000000) mode = 1;
							#ifndef RAPID_NO_AMP
								else mode = 2;
							#else
								else mode = 1;
							#endif

								Array<arrayType, location> res({shape[0], other.shape[1]});

								if (mode == 0)
								{
									// Serial

									uint64 M = shape[0];
									uint64 N = shape[1];
									uint64 K = other.shape[1];

									const arrayType *a = dataStart;
									const arrayType *b = other.dataStart;
									arrayType *c = res.dataStart;

									uint64 i, j, k;
									arrayType tmp;

									for (i = 0; i < M; ++i)
									{
										for (j = 0; j < K; ++j)
										{
											tmp = 0;

											for (k = 0; k < N; ++k)
												tmp += a[k + i * N] * b[j + k * K];

											c[j + i * K] = tmp;
										}
									}
								}
								else if (mode == 1)
								{
									// Parallel

									auto M = (long long) shape[0];
									auto N = (long long) shape[1];
									auto K = (long long) other.shape[1];

									const arrayType *a = dataStart;
									const arrayType *b = other.dataStart;
									arrayType *c = res.dataStart;

									long long i, j, k;
									arrayType tmp;

								#pragma omp parallel for shared(M, N, K, a, b, c) private(i, j, k, tmp) default(none)
									for (i = 0; i < M; ++i)
									{
										for (j = 0; j < K; ++j)
										{
											tmp = 0;

											for (k = 0; k < N; ++k)
												tmp += a[k + i * N] * b[j + k * K];

											c[j + i * K] = tmp;
										}
									}
								}
							#ifndef RAPID_NO_AMP
								else if (mode == 2)
								{
									// Massive parallel

									// Tile size
									static const int TS = 32;

									const auto resizedThis = internal_resized({math::roundUp(shape[0], (uint64) TS),
																			  math::roundUp(shape[1], (uint64) TS)});
									const auto resizedOther = internal_resized({math::roundUp(other.shape[0], (uint64) TS),
																			   math::roundUp(other.shape[1], (uint64) TS)});
									res.internal_resize({math::roundUp(shape[0], (uint64) TS),
														math::roundUp(other.shape[1], (uint64) TS)});

									auto M = (unsigned int) resizedThis.shape[0];
									auto N = (unsigned int) resizedThis.shape[1];
									auto K = (unsigned int) res.shape[1];

									array_view<const arrayType, 2> a(M, N, resizedThis.dataStart);
									array_view<const arrayType, 2> b(N, K, resizedOther.dataStart);
									array_view<arrayType, 2> product(M, K, res.dataStart);

									parallel_for_each(product.extent.tile<TS, TS>(), [=](tiled_index<TS, TS> t_idx) restrict(amp)
									{
										// Get the location of the thread relative to the tile (row, col)
										// and the entire array_view (rowGlobal, colGlobal).
										const int row = t_idx.local[0];
										const int col = t_idx.local[1];
										const int rowGlobal = t_idx.global[0];
										const int colGlobal = t_idx.global[1];
										arrayType sum = 0;

										for (int i = 0; i < M; i += TS)
										{
											tile_static arrayType locA[TS][TS];
											tile_static arrayType locB[TS][TS];
											locA[row][col] = a(rowGlobal, col + i);
											locB[row][col] = b(row + i, colGlobal);

											t_idx.barrier.wait();

											for (int k = 0; k < TS; k++)
												sum += locA[row][k] * locB[k][col];

											t_idx.barrier.wait();
										}

										product[t_idx.global] = sum;
									});

									product.synchronize();

									res.internal_resize({shape[0], other.shape[1]});
								}
							#endif

								return res;
							}
						default:
							{
								std::vector<uint64> resShape = shape;
								resShape[resShape.size() - 2] = shape[shape.size() - 2];
								resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
								Array<arrayType, location> res(resShape);

								for (uint64 i = 0; i < shape[0]; i++)
								{
									res[i] = (operator[](i).dot(other[i]));
								}

								return res;
							}
					}
				#endif
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					switch (dims)
					{
						case 1:
							{
								rapidAssert(shape[0] == other.shape[0], "Invalid shape for array math::product");
								rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array math::product");

								Array<arrayType, location> res({1});
								res.isZeroDim = true;

								cuda::dot(1, shape[0], other.shape[0], dataStart, other.dataStart, res.dataStart);

								return res;
							}
						case 2:
							{
								rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot math::product");
								uint64 size = shape[0] * shape[1] * other.shape[1];

								Array<arrayType, location> res({shape[0], other.shape[1]});

								uint64 m = shape[0];
								uint64 n = shape[1];
								uint64 k = other.shape[1];

								arrayType dotAlpha = 1;
								arrayType dotBeta = 0;

								toColumMajor_inplace();
								other.toColumMajor_inplace();

								cudaSafeCall(cudaDeviceSynchronize());
								cuda::gemm(handle::handle, CUBLAS_OP_N, CUBLAS_OP_N, m, k, n, &dotAlpha, dataStart, m, other.dataStart, n, &dotBeta, res.dataStart, m);

								toRowMajor_inplace();
								other.toRowMajor_inplace();
								res.toRowMajor_inplace();

								return res;
							}
						default:
							{
								std::vector<uint64> resShape = shape;
								resShape[resShape.size() - 2] = shape[shape.size() - 2];
								resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
								Array<arrayType, location> res(resShape);

								for (uint64 i = 0; i < shape[0]; i++)
									res[i] = (*this)[i].dot(other[i]);

								return res;
							}
					}
				}
			#endif
			}

			/// <summary>
			/// Transpose an array and return the result. If the
			/// array is one dimensional, a vector is returned. The
			/// order in which the transpose occurs can be set with
			/// the "axes" parameter
			/// </summary>
			/// <param name="axes"></param>
			/// <returns></returns>
			inline Array<arrayType, location> transposed(const std::vector<uint64> &axes = std::vector<uint64>(), bool dataOnly = false) const
			{
			#ifdef RAPID_DEBUG
				if (!axes.empty())
				{
					if (axes.size() != shape.size())
						message::RapidError("Transpose Error", "Invalid number of axes for array transpose").display();
					for (uint64 i = 0; i < axes.size(); i++)
						if (std::count(axes.begin(), axes.end(), i) != 1)
							message::RapidError("Transpose Error", "Dimension does not appear only once").display();
				}
			#endif

				// Check if a transposition is required
				bool cpy = !axes.empty();
				for (uint64 i = 0; i < axes.size(); i++) if (axes[i] != i) cpy = false;
				if (cpy) return copy();

				std::vector<uint64> newDims;

				if (dataOnly)
				{
					newDims = std::vector<uint64>(shape.begin(), shape.end());
				}
				else
				{
					newDims = std::vector<uint64>(shape.size());
					if (axes.empty())
						for (uint64 i = 0; i < shape.size(); i++)
							newDims[i] = shape[shape.size() - i - 1];
					else
						for (uint64 i = 0; i < shape.size(); i++)
							newDims[i] = shape[axes[i]];
				}

				const uint64 newDimsProd = math::prod(newDims);
				const uint64 shapeProd = math::prod(shape);

				if (location == CPU)
				{
					// Edge case for 1D array
					if (shape.size() == 1 || (axes.size() == 1 && axes[0] == 0))
					{
						auto res = Array<arrayType, location>(newDims);
						memcpy(res.dataStart, dataStart, sizeof(arrayType) * newDimsProd);
						return res;
					}

					if (shape.size() == 2)
					{
						auto res = Array<arrayType, location>(newDims);

						uint64 rows = shape[0];
						uint64 cols = shape[1];

						if (rows * cols < 1000000)
						{
							for (uint64 i = 0; i < rows; i++)
							{
								for (uint64 j = 0; j < cols; j++)
									res.dataStart[i + j * rows] = dataStart[j + i * cols];
							}
						}
						else
						{
							int64 i = 0, j = 0;
							const arrayType *thisData = dataStart;
							arrayType *resData = res.dataStart;
							auto minCols = rapid::math::max(cols, 3) - 3;

						#pragma omp parallel for private(i, j) shared(resData, thisData, minCols, rows, cols) default(none)
							for (i = 0; i < rows; i++)
							{
								for (j = 0; j < minCols; j++)
								{
									int64 p1 = i + j * rows;
									int64 p2 = j + i * cols;

									resData[p1 + 0] = thisData[p2 + 0];
									resData[p1 + 1] = thisData[p2 + 1];
									resData[p1 + 2] = thisData[p2 + 2];
									resData[p1 + 3] = thisData[p2 + 3];
								}

								for (; j < cols; j++)
									resData[i + j * rows] = thisData[+i * cols];
							}
						}

						return res;
					}

					auto res = Array<arrayType, location>(newDims);

					std::vector<uint64> indices(shape.size(), 0);
					std::vector<uint64> indicesRes(shape.size(), 0);

					if (shapeProd < 62000)
					{
						for (int64 i = 0; i < shapeProd; i++)
						{
							if (axes.empty())
								for (int64 j = 0; j < shape.size(); j++)
									indicesRes[j] = indices[shape.size() - j - 1];
							else
								for (int64 j = 0; j < shape.size(); j++)
									indicesRes[j] = indices[axes[j]];

							res.dataStart[imp::dimsToIndex(newDims, indicesRes)] = dataStart[imp::dimsToIndex(shape, indices)];

							indices[shape.size() - 1]++;
							int64 index = shape.size() - 1;

							while (indices[index] >= shape[index] && index > 0)
							{
								indices[index] = 0;
								index--;
								indices[index]++;
							}
						}
					}
					else
					{
						for (int64 i = 0; i < shapeProd; i++)
						{
							if (axes.empty())
								for (int64 j = 0; j < shape.size(); j++)
									indicesRes[j] = indices[shape.size() - j - 1];
							else
								for (int64 j = 0; j < shape.size(); j++)
									indicesRes[j] = indices[axes[j]];

							res.dataStart[imp::dimsToIndex(newDims, indicesRes)] = dataStart[imp::dimsToIndex(shape, indices)];

							indices[shape.size() - 1]++;
							int64 index = shape.size() - 1;

							while (indices[index] >= shape[index] && index > 0)
							{
								indices[index] = 0;
								index--;
								indices[index]++;
							}
						}
					}

					return res;
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					if (shape.size() == 1 || (axes.size() == 1 && axes[0] == 0))
					{
						auto res = Array<arrayType, location>(newDims);
						cudaSafeCall(cudaDeviceSynchronize());
						cudaSafeCall(cudaMemcpy(res.dataStart, dataStart, sizeof(arrayType) * newDimsProd, cudaMemcpyDeviceToDevice));
						return res;
					}
					else if (shape.size() == 2)
					{
						uint64 m = shape[0];
						uint64 n = shape[1];

						auto res = Array<arrayType, location>(newDims);

						static arrayType alpha = 1;
						static arrayType beta = 0;

						cudaSafeCall(cudaDeviceSynchronize());
						cuda::geam(handle::handle,
								   CUBLAS_OP_T, CUBLAS_OP_T,
								   m, n,
								   &alpha,
								   dataStart, n,
								   &beta,
								   dataStart, n,
								   res.dataStart, m);

						return res;
					}
					else
					{
						// auto hostThis = new arrayType[shapeProd];
						// cudaSafeCall(cudaMemcpy(hostThis, dataStart, sizeof(arrayType) * shapeProd, cudaMemcpyDeviceToHost));
						//
						// auto hostRes = new arrayType[shapeProd];
						// cudaSafeCall(cudaMemcpy(hostRes, dataStart, sizeof(arrayType) * shapeProd, cudaMemcpyDeviceToHost));
						//
						// std::vector<uint64> indices(shape.size(), 0);
						// std::vector<uint64> indicesRes(shape.size(), 0);
						//
						// if (shapeProd < 62000)
						// {
						// 	for (uint64 i = 0; i < shapeProd; i++)
						// 	{
						// 		if (axes.empty())
						// 			for (uint64 j = 0; j < shape.size(); j++)
						// 				indicesRes[j] = indices[shape.size() - j - 1];
						// 		else
						// 			for (uint64 j = 0; j < shape.size(); j++)
						// 				indicesRes[j] = indices[axes[j]];
						//
						// 		hostRes[imp::dimsToIndex(newDims, indicesRes)] = hostThis[imp::dimsToIndex(shape, indices)];
						//
						// 		indices[shape.size() - 1]++;
						// 		uint64 index = shape.size() - 1;
						//
						// 		while (indices[index] >= shape[index] && index > 0)
						// 		{
						// 			indices[index] = 0;
						// 			index--;
						// 			indices[index]++;
						// 		}
						// 	}
						// }
						// else
						// {
						// 	for (int64 i = 0; i < shapeProd; i++)
						// 	{
						// 		if (axes.empty())
						// 			for (int64 j = 0; j < shape.size(); j++)
						// 				indicesRes[j] = indices[shape.size() - j - 1];
						// 		else
						// 			for (int64 j = 0; j < shape.size(); j++)
						// 				indicesRes[j] = indices[axes[j]];
						//
						// 		hostRes[imp::dimsToIndex(newDims, indicesRes)] = hostThis[imp::dimsToIndex(shape, indices)];
						//
						// 		indices[shape.size() - 1]++;
						// 		int64 index = shape.size() - 1;
						//
						// 		while (indices[index] >= shape[index] && index > 0)
						// 		{
						// 			indices[index] = 0;
						// 			index--;
						// 			indices[index]++;
						// 		}
						// 	}
						// }
						//
						// auto res = Array<arrayType, location>(newDims);
						// cudaSafeCall(cudaMemcpy(res.dataStart, hostRes, sizeof(arrayType) * shapeProd, cudaMemcpyHostToDevice));
						//
						// delete[] hostThis;
						// delete[] hostRes;
						//
						// return res;

						auto res = Array<arrayType, location>(newDims);

						if (axes.empty())
						{
							std::vector<uint64> tmpAxes(shape.size());
							for (uint64 i = 0; i < shape.size(); i++)
								tmpAxes[i] = shape.size() - i - 1;

							cuda::array_transpose(shape, newDims, tmpAxes, dataStart, res.dataStart);
						}
						else
						{
							cuda::array_transpose(shape, newDims, axes, dataStart, res.dataStart);
						}

						return res;
					}
				}
			#endif

				return Array<arrayType, location>({0, 0});
			}

		#define AUTO ((uint64) -1)

			/// <summary>
			/// Resize an array and return the result. The resulting
			/// array is not linked in any way to the parent array,
			/// so an update in the result will not change a value
			/// in the original array.
			/// </summary>
			/// <param name="newShape"></param>
			/// <returns></returns>
			inline Array<arrayType, location> reshaped(const std::vector<uint64> &newShape) const
			{
				auto tmpNewShape = std::vector<uint64>(newShape.size(), 1);
				auto undefined = (uint64) -1;

				for (uint64 i = 0; i < newShape.size(); i++)
				{
					if (newShape[i] == AUTO)
					{
						if (undefined != AUTO)
							message::RapidError("Resize Error", "Only one AUTO dimension is allowed when resizing").display();
						else
							undefined = i;
					}
					else
					{
						tmpNewShape[i] = newShape[i];
					}
				}

				if (undefined != AUTO)
					tmpNewShape[undefined] = math::prod(shape) / math::prod(tmpNewShape);

				if (math::prod(tmpNewShape) != math::prod(shape))
					message::RapidError("Invalid Shape", "Invalid reshape size. Number of elements differ").display();

				bool zeroDim = false;

				if (isZeroDim && tmpNewShape.size() == 1)
					zeroDim = true;
				else
					zeroDim = false;

				(*originCount)++;
				auto res = Array<arrayType, location>::fromData(tmpNewShape, dataOrigin, dataStart, originCount, zeroDim);

				return res;
			}

			/// <summary>
			/// Resize an array inplace
			/// </summary>
			/// <param name="newShape"></param>
			inline void reshape(const std::vector<uint64> &newShape)
			{
				auto tmpNewShape = std::vector<uint64>(newShape.size(), 1);
				auto undefined = (uint64) -1;

				for (uint64 i = 0; i < newShape.size(); i++)
				{
					if (newShape[i] == AUTO)
					{
						if (undefined != AUTO)
							message::RapidError("Resize Error", "Only one AUTO dimension is allowed when resizing").display();
						else
							undefined = i;
					}
					else
					{
						tmpNewShape[i] = newShape[i];
					}
				}

				if (undefined != AUTO)
					tmpNewShape[undefined] = math::prod(shape) / math::prod(tmpNewShape);

				if (math::prod(tmpNewShape) != math::prod(shape))
					message::RapidError("Invalid Shape", "Invalid reshape size. Number of elements differ").display();

				if (isZeroDim && tmpNewShape.size() == 1)
					isZeroDim = true;
				else
					isZeroDim = false;

				shape = tmpNewShape;
			}

			template<typename Lambda>
			inline Array<arrayType, location> mapped(Lambda func) const
			{
				auto res = Array<arrayType, location>(shape);
				auto size = math::prod(shape);
				auto mode = ExecutionType::SERIAL;

				if (size > 10000) mode = ExecutionType::PARALLEL;

				unaryOpArray(*this, res, mode, func);
				return res;
			}

			/// <summary>
			/// Create an exact copy of an array. The resulting array
			/// is not linked to the parent in any way, so an
			/// </summary>
			/// <returns></returns>
			inline Array<arrayType, location> copy() const
			{
				Array<arrayType, location> res;
				res.isZeroDim = isZeroDim;
				res.shape = shape;
				res.originCount = new uint64;
				*(res.originCount) = 1;

				if (location == CPU)
				{
					res.dataStart = new arrayType[math::prod(shape)];
					memcpy(res.dataStart, dataStart, sizeof(arrayType) * math::prod(shape));
				}
			#ifdef RAPID_CUDA
				else if (location == GPU)
				{
					cudaSafeCall(cudaMalloc(&(res.dataStart), sizeof(arrayType) * math::prod(shape)));
					cudaMemcpy(res.dataStart, dataStart, sizeof(arrayType) * math::prod(shape), cudaMemcpyDeviceToDevice);
				}
			#endif
				res.dataOrigin = res.dataStart;

				return res;
			}

			/// <summary>
			/// Get a string representation of an array
			/// </summary>
			/// <typeparam name="t"></typeparam>
			/// <returns></returns>
			std::string toString(uint64 startDepth = 0) const;
		};

		template<typename t, ArrayLocation loc>
		std::ostream &operator<<(std::ostream &os, const Array<t, loc> &arr)
		{
			return os << arr.toString();
		}

		template<typename t, ArrayLocation loc = CPU>
		inline Array<t, loc> fromScalar(const t &val)
		{
			Array<t, loc> res;

			res.isZeroDim = true;
			res.shape = {1};

			if (loc == CPU)
			{
				res.dataStart = new t[1];
				res.dataStart[0] = val;
			}
		#ifdef RAPID_CUDA
			else
			{
				cudaSafeCall(cudaMalloc(&res.dataStart, sizeof(t)));
				cudaSafeCall(cudaMemcpy(res.dataStart, &val, sizeof(t), cudaMemcpyHostToDevice));
			}
		#endif

			res.dataOrigin = res.dataStart;
			res.originCount = new uint64;
			(*res.originCount) = 1;

			return res;
		}

		template<typename t, ArrayLocation loc = CPU>
		inline Array<t, loc> fromData(const std::initializer_list<t> &data)
		{
		#ifdef RAPID_CUDA
			if (!handle::handleInitialized)
				handle::createHandle();
		#endif

			auto res = Array<t, loc>({data.size()});
			std::vector<t> values;

			for (const auto &val : data)
				values.emplace_back(val);

			if (loc == CPU)
				memcpy(res.dataStart, values.data(), sizeof(t) * values.size());
		#ifdef RAPID_CUDA
			else if (location == GPU)
				cudaSafeCall(cudaMemcpy(res.dataStart, values.data(), sizeof(t) * values.size(), cudaMemcpyHostToDevice));
		#endif

			return res;
		}

	#define imp_temp template<typename t, ArrayLocation loc = CPU>
	#define imp_func_def(x) static inline Array<t, loc> fromData(const x &data)
	#define imp_func_body	auto res = Array<t, loc>(imp::extractShape(data)); \
							    uint64 index = 0; \
								for (const auto &val : data) res[index++] = fromData<t, loc>(val); \
									return res;
	#define L std::initializer_list

		// Up to 20-dimensional array setting from data
		imp_temp imp_func_def(L<L<t>>)
		{
			auto res = Array<t, loc>(imp::extractShape(data));

			uint64 index = 0;
			for (const auto &val : data)
				res[index++] = fromData<t, loc>(val);

			return res;
		}

		imp_temp imp_func_def(L<L<L<t>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<t>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<t>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<t>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<t>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<t>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<t>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>)
		{
			imp_func_body
		}
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>>)
		{
			imp_func_body
		}

	#undef imp_temp
	#undef imp_func_def
	#undef imp_func_body
	#undef L

		template<typename t, ArrayLocation loc = CPU>
		inline Array<t, loc> zeros(const std::vector<uint64> &shape)
		{
			auto res = Array<t, loc>(shape);
			res.fill(0);
			return res;
		}

		template<typename t, ArrayLocation loc = CPU>
		inline Array<t, loc> ones(const std::vector<uint64> &shape)
		{
			auto res = Array<t, loc>(shape);
			res.fill(1);
			return res;
		}

		/// <summary>
		/// Create a new array of the same size and dimensions as
		/// another array, but fill it with zeros.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> zerosLike(const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);
			res.fill((t) 0);
			return res;
		}

		/// <summary>
		/// Create a new array of the same size and dimensions as
		/// another array, but fill it with ones
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> onesLike(const Array<t, loc> &other)
		{
			auto res = Array<t, loc>(other.shape);
			res.fill((t) 1);
			return res;
		}

		/// <summary>
		/// Reverse addition
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> operator+(v val, const Array<t, loc> &other)
		{
			if (loc == CPU)
			{
				auto res = Array<t, loc>(other.shape);
				Array<t, loc>::binaryOpScalarArray((t) val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x + y;
				});

				res.isZeroDim = other.isZeroDim;
				return res;
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::add_scalar_array(math::prod(other.shape), (t) val, other.dataStart, res.dataStart);

				res.isZeroDim = other.isZeroDim;
				return res;
			}
		#endif
		}

		/// <summary>
		/// Reverse subtraction
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> operator-(v val, const Array<t, loc> &other)
		{
			if (loc == CPU)
			{
				auto res = Array<t, loc>(other.shape);
				Array<t, loc>::binaryOpScalarArray((t) val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x - y;
				});

				res.isZeroDim = other.isZeroDim;
				return res;
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::sub_scalar_array(math::prod(other.shape), (t) val, other.dataStart, res.dataStart);

				res.isZeroDim = other.isZeroDim;
				return res;
			}
		#endif
		}

		/// <summary>
		/// Reverse multiplication
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> operator*(v val, const Array<t, loc> &other)
		{
			if (loc == CPU)
			{
				auto res = Array<t, loc>(other.shape);
				Array<t, loc>::binaryOpScalarArray((t) val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x * y;
				});

				res.isZeroDim = other.isZeroDim;
				return res;
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::mul_scalar_array(math::prod(other.shape), (t) val, other.dataStart, 1, res.dataStart, 1);

				res.isZeroDim = other.isZeroDim;
				return res;
			}
		#endif
		}

		/// <summary>
		/// Reverse division
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> operator/(v val, const Array<t, loc> &other)
		{
			if (loc == CPU)
			{
				auto res = Array<t, loc>(other.shape);
				Array<t, loc>::binaryOpScalarArray((t) val, other, res,
												   math::prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL,
												   [](t x, t y)
				{
					return x / y;
				});

				res.isZeroDim = other.isZeroDim;
				return res;
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(other.shape);
				cuda::div_scalar_array(math::prod(other.shape), (t) val, other.dataStart, 1, res.dataStart, 1);
				return res;
			}
		#endif
		}

		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> minimum(const Array<t, loc> &arr, v x)
		{
			if (loc == CPU)
			{
				return arr.mapped([&](t val)
				{
					return val < (t) x ? val : (t) x;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(arr.shape);
				cuda::array_minimum(math::prod(arr.shape), arr.dataStart, 1, x, res.dataStart, 1);
				return res;
			}
		#endif
		}

		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> maximum(const Array<t, loc> &arr, v x)
		{
			if (loc == CPU)
			{
				return arr.mapped([&](t val)
				{
					return val > (t) x ? val : (t) x;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(arr.shape);
				cuda::array_maximum(math::prod(arr.shape), arr.dataStart, 1, x, res.dataStart, 1);
				return res;
			}
		#endif
		}

		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> less(const Array<t, loc> &arr, v x)
		{
			if (loc == CPU)
			{
				return arr.mapped([&](t val)
				{
					return val < (t) x ? 1 : 0;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(arr.shape);
				cuda::array_less(math::prod(arr.shape), arr.dataStart, 1, x, res.dataStart, 1);
				return res;
			}
		#endif
		}

		template<typename t, typename v, ArrayLocation loc>
		inline Array<t, loc> greater(const Array<t, loc> &arr, v x)
		{
			if (loc == CPU)
			{
				return arr.mapped([&](t val)
				{
					return val > (t) x ? 1 : 0;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				auto res = Array<t, loc>(arr.shape);
				cuda::array_greater(math::prod(arr.shape), arr.dataStart, 1, x, res.dataStart, 1);
				return res;
			}
		#endif
		}

		/// <summary>
		/// Sum all of the elements of an array
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> sum(const Array<t, loc> &arr, uint64 axis = (uint64) -1, uint64 depth = 0)
		{
			if (axis == (uint64) -1 || arr.shape.size() == 1)
			{
				if (loc == CPU)
				{
					t res = 0;

					for (uint64 i = 0; i < math::prod(arr.shape); i++)
						res += arr.dataStart[i];
					return Array<t, loc>::fromScalar(res);
				}
			#ifdef RAPID_CUDA
				else if (loc == GPU)
				{
					auto host = new t[math::prod(arr.shape)];
					cudaSafeCall(cudaDeviceSynchronize());
					cudaSafeCall(cudaMemcpy(host, arr.dataStart, sizeof(t) * math::prod(arr.shape), cudaMemcpyDeviceToHost));
					t res = 0;

					for (uint64 i = 0; i < math::prod(arr.shape); i++)
						res += host[i];

					delete[] host;
					return Array<t, loc>::fromScalar(res);
				}
			#endif
			}

			rapidAssert(axis < arr.shape.size(), "Axis '" + std::to_string(axis) +
						"' is out of bounds for array with '" + std::to_string(arr.shape.size()) +
						"' dimensions");

			std::vector<uint64> transposeOrder(arr.shape.size());

			if (depth == 0)
			{
				for (uint64 i = 0; i < axis; i++)
					transposeOrder[i] = i;

				for (uint64 i = axis; i < arr.shape.size() - 1; i++)
					transposeOrder[i] = depth == 0 ? (i + 1) : i;

				transposeOrder[transposeOrder.size() - 1] = axis;
			}
			else
			{
				for (uint64 i = 0; i < arr.shape.size(); i++)
					transposeOrder[i] = i;
			}

			auto fixed = arr.transposed(transposeOrder);

			std::vector<uint64> resShape;
			for (uint64 i = 0; i < transposeOrder.size() - 1; i++)
				resShape.emplace_back(arr.shape[transposeOrder[i]]);

			Array<t, loc> res(resShape);

			for (uint64 outer = 0; outer < res.shape[0]; outer++)
				res[outer] = sum(fixed[outer], math::max(axis, 1) - 1, depth + 1);

			return res;
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> mean(const Array<t, loc> &arr, uint64 axis = (uint64) -1, int depth = 0)
		{
			// Mean of all values
			if (axis == (uint64) -1 || arr.shape.size() == 1)
			{
				return Array<t, loc>(sum(arr) / math::prod(arr.shape));
			}

			rapidAssert(axis < arr.shape.size(), "Axis '" + std::to_string(axis) +
						"' is out of bounds for array with '" + std::to_string(arr.shape.size()) +
						"' dimensions");

			std::vector<uint64> transposeOrder(arr.shape.size());

			if (depth == 0)
			{
				for (uint64 i = 0; i < axis; i++)
					transposeOrder[i] = i;

				for (uint64 i = axis; i < arr.shape.size() - 1; i++)
					transposeOrder[i] = depth == 0 ? (i + 1) : i;

				transposeOrder[transposeOrder.size() - 1] = axis;
			}
			else
			{
				for (uint64 i = 0; i < arr.shape.size(); i++)
					transposeOrder[i] = i;
			}

			auto fixed = arr.transposed(transposeOrder);

			std::vector<uint64> resShape;
			for (uint64 i = 0; i < transposeOrder.size() - 1; i++)
				resShape.emplace_back(arr.shape[transposeOrder[i]]);

			Array<t, loc> res(resShape);

			for (uint64 outer = 0; outer < res.shape[0]; outer++)
				res[outer] = mean(fixed[outer], math::max(axis, 1) - 1, depth + 1);

			return res;
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> abs(const Array<t, loc> &arr)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [](t x)
				{
					return math::abs(x);
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_abs(math::prod(arr.shape), arr.dataStart, 1, result.dataStart, 1);
			}
		#endif

			return result;
		}

		/// <summary>
		/// Calculate the exponent of every value
		/// in an array, and return the result
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> exp(const Array<t, loc> &arr)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [](t x)
				{
					return std::exp(x);
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_exp(math::prod(arr.shape), arr.dataStart, 1, result.dataStart, 1);
			}
		#endif

			return result;
		}

		/// <summary>
		/// Square every element in an array and return
		/// the result
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> square(const Array<t, loc> &arr)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [](t x)
				{
					return x * x;
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_exp(math::prod(arr.shape), arr.dataStart, 1, result.dataStart, 1);
			}
		#endif

			return result;
		}

		/// <summary>
		/// Square root every element in an array
		/// and return the result
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc>
		inline Array<t, loc> sqrt(const Array<t, loc> &arr)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [](t x)
				{
					return std::sqrt(x);
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_square(math::prod(arr.shape), arr.dataStart, 1, result.dataStart, 1);
			}
		#endif

			return result;
		}

		/// <summary>
		/// Raise an array to a power
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <param name="power"></param>
		/// <returns></returns>
		template<typename t, typename p, ArrayLocation loc>
		inline Array<t, loc> pow(const Array<t, loc> &arr, p power)
		{
			Array<t, loc> result(arr.shape);

			if (loc == CPU)
			{
				ExecutionType mode;
				if (math::prod(arr.shape) > 10000)
					mode = ExecutionType::PARALLEL;
				else
					mode = ExecutionType::SERIAL;

				Array<t, loc>::unaryOpArray(arr, result, mode, [=](t x)
				{
					return std::pow(x, (t) power);
				});
			}
		#ifdef RAPID_CUDA
			else if (loc == GPU)
			{
				cuda::array_pow(math::prod(arr.shape), arr.dataStart, 1, (t) power, result.dataStart, 1);
			}
		#endif

			return result;
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> var(const Array<t, loc> &arr, const uint64 axis = (uint64) -1, const uint64 depth = 0)
		{
			// Default variation calculation on flattened array
			if (axis == (uint64) -1 || arr.shape.size() == 1)
				return mean(square(abs(arr - mean(arr))));

			rapidAssert(axis < arr.shape.size(), "Axis '" + std::to_string(axis) +
						"' is out of bounds for array with '" + std::to_string(arr.shape.size()) +
						"' dimensions");

			std::vector<uint64> transposeOrder(arr.shape.size());

			if (depth == 0)
			{
				for (uint64 i = 0; i < axis; i++)
					transposeOrder[i] = i;

				for (uint64 i = axis; i < arr.shape.size() - 1; i++)
					transposeOrder[i] = depth == 0 ? (i + 1) : i;

				transposeOrder[transposeOrder.size() - 1] = axis;
			}
			else
			{
				for (uint64 i = 0; i < arr.shape.size(); i++)
					transposeOrder[i] = i;
			}

			auto fixed = arr.transposed(transposeOrder);

			std::vector<uint64> resShape;
			for (uint64 i = 0; i < transposeOrder.size() - 1; i++)
				resShape.emplace_back(arr.shape[transposeOrder[i]]);

			Array<t, loc> res(resShape);

			for (uint64 outer = 0; outer < res.shape[0]; outer++)
				res[outer] = var(fixed[outer], math::max(axis, 1) - 1, depth + 1);

			return res;
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> sin(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::sin(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> cos(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::cos(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> tan(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::tan(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> asin(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::asin(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> acos(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::acos(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> atan(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::atan(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> sinh(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::sinh(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> cosh(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::cosh(val);
			});
		}

		template<typename t, ArrayLocation loc>
		inline Array<t, loc> tanh(const Array<t, loc> &arr)
		{
			return arr.mapped([](t val)
			{
				return std::tanh(val);
			});
		}

		/// <summary>
		/// Create a vector of a given length where the first element
		/// is "start" and the final element is "end", increasing in
		/// regular increments
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="start"></param>
		/// <param name="end"></param>
		/// <param name="len"></param>
		/// <returns></returns>
		template<typename s, typename e>
		inline Array<typename std::common_type<s, e>::type> linspace(s start, e end, uint64 len)
		{
			using ct = typename std::common_type<s, e>::type;

			Array<ct> result({len});
			result.isZeroDim = len <= 1;

			if (len == 0)
				return result;

			if (len == 1)
			{
				result.dataStart[0] = start;
				return result;
			}

			ct inc = ((ct) end - (ct) start) / (ct) (len - 1);
			for (uint64 i = 0; i < len; i++)
				result.dataStart[i] = (ct) start + (ct) i * inc;

			return result;
		}

		/// <summary>
		/// Create a vector of a specified type, where the values
		/// increase/decrease linearly between a start and end
		/// point by a specified amount
		/// </summary>
		/// <typeparam name="s"></typeparam>
		/// <typeparam name="e"></typeparam>
		/// <typeparam name="t"></typeparam>
		/// <param name="start"></param>
		/// <param name="end"></param>
		/// <param name="inc"></param>
		/// <returns></returns>
		template<typename s, typename e, typename iT = s, ArrayLocation loc = CPU>
		inline Array<typename std::common_type<s, e>::type, loc> arange(s start, e end, iT inc = 1)
		{
			using ct = typename std::common_type<s, e>::type;

			auto len = (uint64) ceil(math::abs((ct) end - (ct) start) / (ct) inc);
			auto res = Array<typename std::common_type<s, e>::type, loc>({len});
			for (uint64 i = 0; i < len; i++)
				res[i] = (ct) start + (ct) inc * (ct) i;
			return res;
		}

		/// <summary>
		/// Create a vector of a specified type, where the values
		/// increase/decrease linearly between a start and end
		/// point by an specified
		/// </summary>
		/// <typeparam name="e"></typeparam>
		/// <param name="end"></param>
		/// <returns></returns>
		template<typename e, ArrayLocation loc = CPU>
		inline Array<e, loc> arange(e end)
		{
			return arange((e) 0, end, (e) 1);
		}

		/// <summary>
		/// Create a 3D array from two vectors, where the first element
		/// is vector A in row format, and the second element is vector
		/// B in column format.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc = CPU>
		inline Array<t, loc> meshgrid(const Array<t, loc> &a, const Array<t, loc> &b)
		{
			rapidAssert(a.shape.size() == 1 && b.shape.size() == 1, "Invalid size for meshgrid. Must be a 1D array");
			Array<t, loc> result({2, b.shape[0], a.shape[0]});

			if (math::prod(result.shape) < 10000)
			{
				for (int64 i = 0; i < b.shape[0]; i++)
					for (int64 j = 0; j < a.shape[0]; j++)
						result.setVal({(int64) 0, i, j}, a.accessVal({j}));

				for (int64 i = 0; i < b.shape[0]; i++)
					for (int64 j = 0; j < a.shape[0]; j++)
						result.setVal({(int64) 1, i, j}, b.accessVal({i}));
			}
			else
			{
			#pragma omp parallel for
				for (int64 i = 0; i < b.shape[0]; i++)
					for (int64 j = 0; j < a.shape[0]; j++)
						result.setVal({(int64) 0, i, j}, a.accessVal({j}));

			#pragma omp parallel for
				for (int64 i = 0; i < b.shape[0]; i++)
					for (int64 j = 0; j < a.shape[0]; j++)
						result.setVal({(int64) 1, i, j}, b.accessVal({i}));
			}

			return result;
		}

		/// <summary>
		/// Return a gaussian matrix with the given rows, columns and
		/// standard deviation.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="r"></param>
		/// <param name="c"></param>
		/// <param name="sigma"></param>
		/// <returns></returns>
		template<typename t, ArrayLocation loc = CPU>
		inline Array<t, loc> gaussian(uint64 r, uint64 c, t sigma)
		{
			t rows = (t) r;
			t cols = (t) c;

			auto ax = linspace<t>(-(rows - 1) / 2., (rows - 1) / 2., r);
			auto ay = linspace<t>(-(cols - 1) / 2., (cols - 1) / 2., c);
			auto mesh = meshgrid(ay, ax);
			auto xx = mesh[0];
			auto yy = mesh[1];

			auto kernel = exp(-0.5 * (square(xx) + square(yy)) / (sigma * sigma));
			return kernel / sum(kernel);
		}

		/// <summary>
		/// Cast an array from one type to another. This makes a copy of the array,
		/// and therefore altering a value in one will not cause an update in the
		/// other.
		/// </summary>
		/// <typeparam name="res"></typeparam>
		/// <typeparam name="src"></typeparam>
		/// <param name="src"></param>
		/// <returns></returns>
		template<typename resT, typename srcT, ArrayLocation srcL, ArrayLocation loc = CPU>
		inline Array<resT, loc> cast(const Array<srcT, srcL> &src)
		{
			Array<resT, loc> res(src.shape);

			if (loc == CPU && srcL == CPU)
			{
				if (math::prod(src.shape) < 10000)
				{
					for (int64 i = 0; i < math::prod(src.shape); i++)
						res.dataStart[i] = (resT) src.dataStart[i];
				}
				else
				{
				#pragma omp parallel for
					for (int64 i = 0; i < math::prod(src.shape); i++)
						res.dataStart[i] = (resT) src.dataStart[i];
				}
			}
		#ifdef RAPID_CUDA
			else if (loc == CPU && srcL == GPU)
			{
				if (math::prod(src.shape) < 10000)
				{
					srcT srcVal = 0;
					for (int64 i = 0; i < math::prod(src.shape); i++)
					{
						cudaSafeCall(cudaMemcpy(&srcVal, src.dataStart + i, sizeof(srcT), cudaMemcpyDeviceToHost));
						res.dataStart[i] = (resT) srcVal;
					}
				}
				else
				{
					srcT srcVal = 0;
				#pragma omp parallel for private(srcVal)
					for (int64 i = 0; i < math::prod(src.shape); i++)
					{
						cudaSafeCall(cudaMemcpy(&srcVal, src.dataStart + i, sizeof(srcT), cudaMemcpyDeviceToHost));
						res.dataStart[i] = (resT) srcVal;
					}
				}
			}
			else if (loc == GPU && srcL == CPU)
			{
				if (math::prod(src.shape) < 10000)
				{
					resT srcVal = 0;
					for (int64 i = 0; i < math::prod(src.shape); i++)
					{
						srcVal = (resT) src.dataStart[i];
						cudaSafeCall(cudaMemcpy(res.dataStart + i, &srcVal, sizeof(resT), cudaMemcpyHostToDevice));
					}
				}
				else
				{
					resT srcVal = 0;
				#pragma omp parallel for private(srcVal)
					for (int64 i = 0; i < math::prod(src.shape); i++)
					{
						srcVal = (resT) src.dataStart[i];
						cudaSafeCall(cudaMemcpy(res.dataStart + i, &srcVal, sizeof(resT), cudaMemcpyHostToDevice));
					}
				}
			}
			else if (loc == GPU && srcL == CPU)
			{
				cuda::array_cast((unsigned int) math::prod(src.shape), src.dataStart, 1, res.dataStart, 1);
			}
		#endif

			return res;
		}
	}
}
