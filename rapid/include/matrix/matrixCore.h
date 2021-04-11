#pragma once

#include "../internal.h"
#include "matrixArrayView.h"
#include "../IO/createDir.h"
#include "../messageBox.h"

#ifndef RAPID_NO_AMP
using namespace concurrency;
#endif

constexpr int RAPID_MATH_MODE_SERIAL = 0;
constexpr int RAPID_MATH_MODE_PARALLEL = 1;
constexpr int RAPID_MATH_MODE_MASSIVE_PARALLEL = 2;

constexpr int RAPID_MATH_OP_MATRIX_MATRIX_ADDITION = 0;
constexpr int RAPID_MATH_OP_MATRIX_MATRIX_SUBTRACTION = 1;
constexpr int RAPID_MATH_OP_MATRIX_MATRIX_MULTIPLICATION = 2;
constexpr int RAPID_MATH_OP_MATRIX_MATRIX_DIVISION = 3;

constexpr int RAPID_MATH_OP_MATRIX_SCALAR_ADDITION = 4;
constexpr int RAPID_MATH_OP_MATRIX_SCALAR_SUBTRACTION = 5;
constexpr int RAPID_MATH_OP_MATRIX_SCALAR_MULTIPLICATION = 6;
constexpr int RAPID_MATH_OP_MATRIX_SCALAR_DIVISION = 7;

constexpr int RAPID_MATH_OP_MATRIX_UNARY = 8;

constexpr int RAPID_MATH_OP_MATRIX_TRANSPOSE = 9;
constexpr int RAPID_MATH_OP_MATRIX_PRODUCT = 10;

namespace rapid
{
	namespace matrix
	{
		typedef struct MatrixSize
		{
			size_t rows;
			size_t cols;

			size_t operator[](size_t index) const
			{
				if (index == 0)
					return rows;
				if (index == 1)
					return cols;

				message::RapidError("Index Error", "Index out of range for matrix size object. Maximum index is 1").display();
				return -1;
			}

			size_t operator()(size_t index) const
			{
				if (index == 0)
					return rows;
				else if (index == 1)
					return cols;
				message::RapidError("Index Error", "Index out of range for matrix size object. Maximum index is 1").display();
				return -1;
			}
		};

		template<typename dataType>
		class Matrix
		{
		private:
			size_t rows;
			size_t cols;
			std::vector<dataType> data;

			// Operation method evaluator
			inline static int evalOperationMode(size_t M, size_t N, size_t K, int op)
			{
				// Matrix-matrix and matrix-scalar arithmetic operators
				if (op >= RAPID_MATH_OP_MATRIX_MATRIX_ADDITION && op <= RAPID_MATH_OP_MATRIX_SCALAR_DIVISION)
				{
					if (M * N < 600 * 600)
						return RAPID_MATH_MODE_SERIAL;
					return RAPID_MATH_MODE_PARALLEL;
				}

				// Matrix unary operation
				if (op == RAPID_MATH_OP_MATRIX_UNARY)
				{
					if (M * N < 600 * 600)
						return RAPID_MATH_MODE_SERIAL;
					return RAPID_MATH_MODE_PARALLEL;
				}

				// Transposition
				if (op == RAPID_MATH_OP_MATRIX_TRANSPOSE)
				{
					if (M * N < 550 * 500)
						return RAPID_MATH_MODE_SERIAL;
					return RAPID_MATH_MODE_PARALLEL;
				}

				// Matrix product
				if (op == RAPID_MATH_OP_MATRIX_PRODUCT)
				{
				#ifndef RAPID_NO_AMP
					if (M * N * K >= 400 * 400 * 400)
						return RAPID_MATH_MODE_MASSIVE_PARALLEL;
				#endif
					// if (M * N * K >= 50 * 50 * 50)
					if (M * N * K >= 20 * 20 * 20)
						return RAPID_MATH_MODE_PARALLEL;
					return RAPID_MATH_MODE_SERIAL;
				}

				return -1;
			}

			template<typename Lambda>
			inline static void matrixMatrixBinary(const Matrix<dataType> &a, const Matrix<dataType> &b, Matrix<dataType> &c, int mode, Lambda func)
			{
				rapidAssert(a.rows == b.rows && a.cols == b.cols, "Invalid operands for matrix addition");
				rapidAssert(a.rows == c.rows && a.cols == c.cols, "Invalid result for matrix addition");

				if (mode == RAPID_MATH_MODE_SERIAL)
				{
					// Serial addition
					size_t index = 0;

					if (a.rows * a.cols > 3)
					{
						for (index = 0; index < a.rows * a.cols - 3; index += 4)
						{
							c.data[index + 0] = func(a.data[index + 0], b.data[index + 0]);
							c.data[index + 1] = func(a.data[index + 1], b.data[index + 1]);
							c.data[index + 2] = func(a.data[index + 2], b.data[index + 2]);
							c.data[index + 3] = func(a.data[index + 3], b.data[index + 3]);
						}
					}

					for (; index < a.rows * a.cols; index++)
						c.data[index] = func(a.data[index], b.data[index]);
				}
				else if (mode == RAPID_MATH_MODE_PARALLEL)
				{
					// Concurrent addition on CPU
					long index = 0;
					long concurrentRows = a.rows;
					long concurrentCols = a.cols;

				#pragma omp parallel for shared(concurrentRows, concurrentCols, a, b, c) private(index) default(none)
					for (index = 0; index < concurrentRows * concurrentCols; ++index)
						c.data[index] = func(a.data[index], b.data[index]);
				}
			}

			template<typename Lambda>
			inline static void matrixScalarBinary(const Matrix<dataType> &a, const dataType &b, Matrix<dataType> &c, int mode, Lambda func)
			{
				rapidAssert(a.rows == c.rows && a.cols == c.cols, "Invalid result for matrix addition");

				if (mode == RAPID_MATH_MODE_SERIAL)
				{
					// Serial addition
					size_t index = 0;

					if (a.rows * a.cols > 3)
					{
						for (index = 0; index < a.rows * a.cols - 3; index += 4)
						{
							c.data[index + 0] = func(a.data[index + 0], b);
							c.data[index + 1] = func(a.data[index + 1], b);
							c.data[index + 2] = func(a.data[index + 2], b);
							c.data[index + 3] = func(a.data[index + 3], b);
						}
					}

					for (; index < a.rows * a.cols; index++)
						c.data[index] = func(a.data[index], b);
				}
				else if (mode == RAPID_MATH_MODE_PARALLEL)
				{
					// Concurrent addition on CPU
					long index = 0;
					long concurrentRows = a.rows;
					long concurrentCols = a.cols;

				#pragma omp parallel for shared(concurrentRows, concurrentCols, a, b, c) private(index) default(none)
					for (index = 0; index < concurrentRows * concurrentCols; ++index)
						c.data[index] = func(a.data[index], b);
				}
			}

			template<typename Lambda>
			inline static void matrixMatrixUnary(const Matrix<dataType> &a, Matrix<dataType> &c, int mode, Lambda func)
			{
				rapidAssert(a.rows == c.rows && a.cols == c.cols, "Invalid result for matrix addition");

				if (mode == RAPID_MATH_MODE_SERIAL)
				{
					// Serial addition
					size_t index = 0;

					if (a.rows * a.cols > 3)
					{
						for (index = 0; index < a.rows * a.cols - 3; index += 4)
						{
							c.data[index + 0] = func(a.data[index + 0]);
							c.data[index + 1] = func(a.data[index + 1]);
							c.data[index + 2] = func(a.data[index + 2]);
							c.data[index + 3] = func(a.data[index + 3]);
						}
					}

					for (; index < a.rows * a.cols; index++)
						c.data[index] = func(a.data[index]);
				}
				else if (mode == RAPID_MATH_MODE_PARALLEL)
				{
					// Concurrent addition on CPU
					long index = 0;
					long concurrentRows = a.rows;
					long concurrentCols = a.cols;

				#pragma omp parallel for shared(concurrentRows, concurrentCols, a, c) private(index) default(none)
					for (index = 0; index < concurrentRows * concurrentCols; ++index)
						c.data[index] = func(a.data[index]);
				}
			}

			inline void checkNan(const std::string &msg = "NaN detected") const
			{
				const dataType *__restrict tmp = data.data();
				for (size_t i = 0; i < rows * cols; i++)
					if (tmp[i] != tmp[i])
						message::RapidError("NaN Detected", msg).display();
			}

		public:
			// Default constructor
			Matrix() : rows(0), cols(0)
			{};

			// From rows
			Matrix(size_t matrixRows)
			{
				rows = matrixRows;
				cols = 1;

				data = std::vector<dataType>(rows * cols, 0);
			}

			// From rows and columns
			Matrix(size_t matrixRows, size_t matrixCols)
			{
				rows = matrixRows;
				cols = matrixCols;

				data = std::vector<dataType>(rows * cols, 0);
			}

			// From rows, columns and specified fill value
			Matrix(size_t matrixRows, size_t matrixCols, const dataType &fill)
			{
				rows = matrixRows;
				cols = matrixCols;

				data = std::vector<dataType>(rows * cols, fill);

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in constructor");
			#endif
			}

			// Copy constructor
			Matrix(const Matrix<dataType> &mat)
			{
				rows = mat.rows;
				cols = mat.cols;

				data = mat.data;
			}

			Matrix(const std::initializer_list<std::initializer_list<dataType>> &matrixData)
			{
				rows = matrixData.size();
				cols = matrixData.begin()->size();

				data = std::vector<dataType>(rows * cols);

				for (size_t i = 0; i < rows; i++)
					for (size_t j = 0; j < cols; j++)
						data[j + i * cols] = *((matrixData.begin() + i)->begin() + j);

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in constructor");
			#endif
			}

			// Copy operator
			Matrix<dataType> &operator=(const Matrix<dataType> &other)
			{
				if (&other != this)
				{
					rows = other.rows;
					cols = other.cols;

					data = other.data;
				}

				return *this;
			}

			static Matrix<dataType> random(const size_t &matrixRows, const size_t &matrixCols = 1, const dataType &minVal = -1, const dataType &maxVal = 1)
			{
				auto res = Matrix<dataType>(matrixRows, matrixCols);

				res.map([=](dataType x)
				{
					return rapid::random<dataType>(minVal, maxVal);
				});

				return res;
			}

			static Matrix<dataType> zeros(const size_t &matrixRows)
			{
				auto res = Matrix<dataType>(matrixRows, 1);

				res.map([](dataType x)
				{
					return 0;
				});

				return res;
			}

			static Matrix<dataType> zeros(const size_t &matrixRows, const size_t &matrixCols)
			{
				auto res = Matrix<dataType>(matrixRows, matrixCols);

				res.map([](dataType x)
				{
					return 0;
				});

				return res;
			}

			static Matrix<dataType> zerosLike(const Matrix<dataType> &input)
			{
				auto res = Matrix<dataType>(input.rows, input.cols);

				res.map([](dataType x)
				{
					return 0;
				});

				return res;
			}

			static Matrix<dataType> ones(const size_t &matrixRows)
			{
				auto res = Matrix<dataType>(matrixRows, 1);

				res.map([](dataType x)
				{
					return 1;
				});

				return res;
			}

			static Matrix<dataType> ones(const size_t &matrixRows, const size_t &matrixCols)
			{
				auto res = Matrix<dataType>(matrixRows, matrixCols);

				res.map([](dataType x)
				{
					return 1;
				});

				return res;
			}

			static Matrix<dataType> ones_like(const Matrix<dataType> &input)
			{
				auto res = Matrix<dataType>(input.rows, input.cols);

				res.map([](dataType x)
				{
					return 1;
				});

				return res;
			}

			static Matrix<dataType> truncatedNormal(const size_t &matrixRows, const size_t &matrixCols, const dataType &mean = 0, const dataType &stddev = 1)
			{
				auto res = Matrix<dataType>::random(matrixRows, matrixCols);
				res *= stddev / res.stddev();
				res += mean - res.mean();

				return res;
			}

			// Fill the matrix with a single value
			inline void fill(const dataType &val)
			{
				for (size_t i = 0; i < rows * cols; i++)
					data[i] = val;

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in constructor");
			#endif
			}

			// Get a row of the matrix
			inline ArrayView<dataType> &operator[](const size_t &index)
			{
				rapidAssert(index < rows, "List index out of range");

				ArrayView<dataType> res(&data.data()[index * cols], cols);

				return res;
			}

			// Get a row of the matrix
			inline ArrayView<const dataType> &operator[](const size_t &index) const
			{
				rapidAssert(index < rows, "List index out of range");

				ArrayView<const dataType> res(&data.data()[index * cols], cols);

				return res;
			}

			// Matrix-matrix addition
			inline Matrix<dataType> operator+(const Matrix<dataType> &other) const
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix addition");

				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixMatrixBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a + b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-matrix addition");
			#endif

				return res;
			}

			// Matrix-matrix subtraction
			inline Matrix<dataType> operator-(const Matrix<dataType> &other) const
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix subtraction");

				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixMatrixBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_SUBTRACTION), [](dataType a, dataType b)
				{
					return a - b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-matrix subtraction");
			#endif

				return res;
			}

			// Matrix-matrix multiplication
			inline Matrix<dataType> operator*(const Matrix<dataType> &other) const
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix multiplication");

				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixMatrixBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_MULTIPLICATION), [](dataType a, dataType b)
				{
					return a * b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-matrix multiplication");
			#endif

				return res;
			}

			// Matrix-matrix division
			inline Matrix<dataType> operator/(const Matrix<dataType> &other) const
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix division");

				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixMatrixBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_DIVISION), [](dataType a, dataType b)
				{
					return a / b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-matrix division");
			#endif

				return res;
			}

			// Matrix-scalar addition
			inline Matrix<dataType> operator+(const dataType &other) const
			{
				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixScalarBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_SCALAR_ADDITION), [](dataType a, dataType b)
				{
					return a + b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-scalar addition");
			#endif

				return res;
			}

			// Matrix-scalar subtraction
			inline Matrix<dataType> operator-(const dataType &other) const
			{
				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixScalarBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_SCALAR_SUBTRACTION), [](dataType a, dataType b)
				{
					return a - b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-scalar subtraction");
			#endif

				return res;
			}

			// Matrix-scalar multiplication
			inline Matrix<dataType> operator*(const dataType &other) const
			{
				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixScalarBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_SCALAR_MULTIPLICATION), [](dataType a, dataType b)
				{
					return a * b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-scalar multiplication");
			#endif

				return res;
			}

			// Matrix-scalar division
			inline Matrix<dataType> operator/(const dataType &other) const
			{
				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixScalarBinary(*this, other, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_SCALAR_DIVISION), [](dataType a, dataType b)
				{
					return a / b;
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-scalar division");
			#endif

				return res;
			}

			// Matrix-matrix inplace addition
			inline void operator+=(const Matrix<dataType> &other)
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix inplace addition");

				Matrix<dataType>::matrixMatrixBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a + b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-matrix inplace addition");
			#endif
			}

			// Matrix-matrix inplace subtraction
			inline void operator-=(const Matrix<dataType> &other)
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix inplace subtraction");

				Matrix<dataType>::matrixMatrixBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a - b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-matrix inplace subtraction");
			#endif
			}

			// Matrix-matrix inplace multiplication
			inline void operator*=(const Matrix<dataType> &other)
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix inplace multiplication");

				Matrix<dataType>::matrixMatrixBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a * b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-matrix inplace multiplication");
			#endif
			}

			// Matrix-matrix inplace division
			inline void operator/=(const Matrix<dataType> &other)
			{
				rapidAssert(rows == other.rows && cols == other.cols, "Matrices must have the same dimensions for matrix-matrix inplace division");

				Matrix<dataType>::matrixMatrixBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a / b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-matrix inplace division");
			#endif
			}

			// Matrix-scalar inplace addition
			inline void operator+=(const dataType &other)
			{
				Matrix<dataType>::matrixScalarBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a + b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-scalar inplace addition");
			#endif
			}

			// Matrix-scalar inplace subtraction
			inline void operator-=(const dataType &other)
			{
				Matrix<dataType>::matrixScalarBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a - b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-scalar inplace subtraction");
			#endif
			}

			// Matrix-scalar inplace multiplication
			inline void operator*=(const dataType &other)
			{
				Matrix<dataType>::matrixScalarBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a * b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-scalar inplace multiplication");
			#endif
			}

			// Matrix-scalar inplace division
			inline void operator/=(const dataType &other)
			{
				Matrix<dataType>::matrixScalarBinary(*this, other, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_MATRIX_ADDITION), [](dataType a, dataType b)
				{
					return a / b;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix-scalar inplace division");
			#endif
			}

			// Unary negation
			inline Matrix<dataType> operator-()
			{
				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixMatrixUnary(*this, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_SCALAR_DIVISION), [](dataType x)
				{
					return -x;
				});

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in unary negation");
			#endif

				return res;
			}

			inline Matrix<dataType> transposed() const
			{
				auto res = Matrix<dataType>(cols, rows);

				if (rows == 1 || cols == 1)
				{
					// Simply swap rows and cols
					memcpy(res.data.data(), data.data(), sizeof(dataType) * rows * cols);
					return res;
				}

				auto mode = Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_TRANSPOSE);

				if (mode == RAPID_MATH_MODE_SERIAL)
				{
					// Serial transposition of the matrix

					size_t row = 0;
					size_t col = 0;

					const dataType *__restrict thisData = data.data();
					dataType *__restrict resData = res.data.data();

					if (cols > 3)
					{
						for (row = 0; row < rows; row++)
						{
							for (col = 0; col < cols - 3; col += 4)
							{
								resData[row + col * rows + 0] = thisData[col + row * cols + 0];
								resData[row + col * rows + 1] = thisData[col + row * cols + 1];
								resData[row + col * rows + 2] = thisData[col + row * cols + 2];
								resData[row + col * rows + 3] = thisData[col + row * cols + 3];
							}
						}
					}

					for (; row < rows; row++)
						for (col = 0; col < cols; col++)
							resData[row + col * rows] = thisData[col + row * cols];
				}
				else if (mode == RAPID_MATH_MODE_PARALLEL)
				{
					// Parallel transposition of the matrix

					long long row = 0;
					long long col = 0;
					long long tempRows = rows;
					long long tempCols = cols;

					const dataType *__restrict tempData = data.data();
					dataType *__restrict resData = res.data.data();

				#pragma omp parallel for shared(res, tempData, resData, tempRows, tempCols) private(row, col) default(none)
					for (row = 0; row < tempRows; row++)
						for (col = 0; col < tempCols; col++)
							resData[row + col * rows] = tempData[col + row * cols];
				}

				return res;
			}

			inline void transpose()
			{
				*this = transposed();
			}

			// Everything else
			inline Matrix<dataType> dot(const Matrix<dataType> &other) const
			{
				rapidAssert(cols == other.rows, "Invalid size for matrix dot product");

				Matrix<dataType> res(rows, other.cols);

				auto mode = Matrix<dataType>::evalOperationMode(rows, cols, other.cols, RAPID_MATH_OP_MATRIX_PRODUCT);

				if (mode == 0)
				{
					// Serial

					size_t M = rows;
					size_t N = cols;
					size_t K = other.cols;

					const dataType *__restrict a = data.data();
					const dataType *__restrict b = other.data.data();
					dataType *__restrict c = res.data.data();

					size_t i, j, k;
					dataType tmp;

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

					auto M = (long long) rows;
					auto N = (long long) cols;
					auto K = (long long) other.cols;

					const dataType *__restrict a = data.data();
					const dataType *__restrict b = other.data.data();
					dataType *__restrict c = res.data.data();

					long long i, j, k;
					dataType tmp;

				#pragma omp parallel for shared(M, N, K, a, b, c) private(i, j, k, tmp) default(none) num_threads(16)
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

					const auto resizedThis = resized(rapid::roundUp(rows, (size_t) TS), rapid::roundUp(cols, (size_t) TS));
					const auto resizedOther = resized(rapid::roundUp(other.rows, (size_t) TS), rapid::roundUp(other.cols, (size_t) TS));
					res.resize(rapid::roundUp(rows, (size_t) TS), rapid::roundUp(other.cols, (size_t) TS));

					auto M = (unsigned int) resizedThis.rows;
					auto N = (unsigned int) resizedThis.cols;
					auto K = (unsigned int) res.cols;

					array_view<const dataType, 2> a(M, N, resizedThis.data);
					array_view<const dataType, 2> b(N, K, resizedOther.data);
					array_view<dataType, 2> product(M, K, res.data);

					parallel_for_each(product.extent.tile<TS, TS>(), [=](tiled_index<TS, TS> t_idx) restrict(amp)
					{
						// Get the location of the thread relative to the tile (row, col)
						// and the entire array_view (rowGlobal, colGlobal).
						const int row = t_idx.local[0];
						const int col = t_idx.local[1];
						const int rowGlobal = t_idx.global[0];
						const int colGlobal = t_idx.global[1];
						dataType sum = 0;

						for (int i = 0; i < M; i += TS)
						{
							tile_static dataType locA[TS][TS];
							tile_static dataType locB[TS][TS];
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

					res.resize(rows, other.cols);
				}
			#endif

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix dot product");
			#endif

				return res;
			}

			inline Matrix<dataType> pow(const Matrix<dataType> &exp) const
			{
				auto res = Matrix<dataType>::zerosLike(*this);

				for (size_t i = 0; i < rows * cols; i++)
					res.data[0] = pow(data[i], exp.data[i]);

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-matrix pow");
			#endif

				return res;
			}

			inline Matrix<dataType> pow(const dataType &exp) const
			{
				auto res = *this;

				res.map([=](dataType x)
				{
					return std::pow(x, exp);
				});

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix-scalar pow");
			#endif

				return res;
			}

			inline Matrix<dataType> resized(size_t newRows, size_t newCols) const
			{
				Matrix<dataType> res(newRows, newCols);

				auto resData = res.data.data();
				auto thisData = data.data();

				for (size_t i = 0; i < rapid::min(rows, newRows); i++)
					memcpy(&resData[i * newCols], &thisData[i * cols], sizeof(dataType) * rapid::min(cols, newCols));

				return res;
			}

			inline void resize(size_t newRows, size_t newCols)
			{
				*this = resized(newRows, newCols);
			}

			static inline Matrix<dataType> vstack(const Matrix &a, const Matrix &b)
			{
				rapidAssert(a.cols == b.cols, "Invalid size for vertical stack");

				auto res = Matrix<dataType>(a.rows + b.rows, a.cols);

				const dataType *__restrict aData = a.data.data();
				const dataType *__restrict bData = b.data.data();
				dataType *__restrict resData = res.data.data();

				// Copy the data over
				memcpy(resData, aData, sizeof(dataType) * a.rows * a.cols);
				memcpy(resData + a.rows * a.cols, bData, sizeof(dataType) * b.rows * b.cols);

				return res;
			}

			static inline Matrix<dataType> hstack(const Matrix &a, const Matrix &b)
			{
				rapidAssert(a.rows == b.rows, "Invalid size for horizontal stack");

				auto res = Matrix<dataType>(a.rows, a.cols + b.cols);

				const dataType *__restrict aData = a.data.data();
				const dataType *__restrict bData = b.data.data();
				dataType *__restrict resData = res.data.data();

				// Copy the data over
				for (size_t i = 0; i < a.rows; i++)
				{
					memcpy(resData + res.cols * i, aData + a.cols * i, sizeof(dataType) * a.cols); // Copy segment of A
					memcpy(resData + res.cols * i + a.cols, bData + b.cols * i, sizeof(dataType) * b.cols); // Copy segment of B
				}

				return res;
			}

			inline Matrix<dataType> subMatrix(size_t splitStart, size_t splitEnd = -1) const
			{
				if (splitEnd == -1)
					splitEnd = rows;

				rapidAssert(splitStart < rows, "Split start index out of range");
				rapidAssert(splitEnd <= rows, "Split end index out of range");
				rapidAssert(splitStart < splitEnd, "Invalid matrix split options");

				auto res = Matrix<dataType>(splitEnd - splitStart, cols);

				const dataType *__restrict thisData = data.data();
				dataType *__restrict resData = res.data.data();

				memcpy(resData, thisData + splitStart * cols, sizeof(dataType) * res.rows * res.cols);

				return res;
			}

			// Compute the outer product of two matrices
			static inline Matrix<dataType> outer(const Matrix<dataType> &a, const Matrix<dataType> &b)
			{
				auto res = Matrix<dataType>(a.rows * a.cols, b.rows * b.cols);

				const dataType *__restrict aData = a.data.data();
				const dataType *__restrict bData = b.data.data();
				dataType *__restrict resData = res.data.data();

				for (size_t i = 0; i < res.rows; i++)
				{
					for (size_t j = 0; j < res.cols; j++)
					{
						resData[j + i * res.cols] = aData[i] * bData[j];
					}
				}

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix outer product");
			#endif

				return res;
			}

			// Limit the values in a matrix to a particular value
			inline Matrix<dataType> clip(dataType lowerBound, dataType upperBound)
			{
				auto res = *this;
				res.map([=](dataType x)
				{
					return x < lowerBound ? lowerBound : (x > upperBound ? upperBound : x);
				});
				return res;
			}

			// Return a matrix mapped by a particular function
			template<typename Lambda>
			inline Matrix<dataType> mapped(Lambda func)
			{
				auto res = Matrix<dataType>(rows, cols);
				Matrix<dataType>::matrixMatrixUnary(*this, res, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_UNARY), func);

			#ifdef RAPID_CHECK_NAN
				res.checkNan("NaN detected in matrix map");
			#endif

				return res;
			}

			// Map the matrix by a particular function
			template<typename Lambda>
			inline void map(Lambda func)
			{
				Matrix<dataType>::matrixMatrixUnary(*this, *this, Matrix<dataType>::evalOperationMode(rows, cols, 0, RAPID_MATH_OP_MATRIX_UNARY), func);

			#ifdef RAPID_CHECK_NAN
				checkNan("NaN detected in matrix inplace map");
			#endif
			}

			inline dataType largest() const
			{
				auto big = (dataType) -INFINITY;

				for (const auto &val : data)
					if (val > big)
						big = val;

				return big;
			}

			inline dataType smallest() const
			{
				auto smol = (dataType) INFINITY;

				for (const auto &val : data)
					if (val < smol)
						smol = val;

				return smol;
			}

			inline dataType sum() const
			{
				dataType total = 0;

				for (const auto &val : data)
					total += val;

			#ifdef RAPID_CHECK_NAN
				if (total != total)
					message::RapidError("NaN Detected", "NaN detected in matrix sum");
			#endif

				return total;
			}

			inline dataType mean() const
			{
				rapidAssert(rows * cols != 0, "Matrix cannot be 0x0 for matrix mean");

				return sum() / (rows * cols);
			}

			inline dataType stddev() const
			{
				auto meanAverage = mean();
				Matrix<dataType> variance(rows * cols, 1, 0);

				size_t index = 0;
				for (const auto &val : data)
				{
					variance[index][0] = (val - meanAverage) * (val - meanAverage);

				#ifdef RAPID_CHECK_NAN
					if (variance[index] != variance[index])
						message::RapidError("NaN detected", "NaN detected in matrix-scalar inplace addition");
				#endif

					index++;
				}

				return sqrt(variance.mean());
			}

			inline MatrixSize size() const
			{
				return {rows, cols};
			}

			// Convert to std::string
			std::string toString() const
			{
				std::string res;

				for (size_t i = 0; i < rows; i++)
				{
					for (size_t j = 0; j < cols; j++)
					{
						res += rapidCast<std::string>(data[j + i * cols]);

						if (j < cols - 1)
							res += ", ";
					}
					if (i < rows - 1)
						res += "\n";
				}

				return res;
			}

			inline void saveToFile(const std::string dir, bool verbose = false)
			{
				io::createDirectory(dir.substr(0, dir.find_last_of("/")));

				std::fstream file;
				file.open(dir, std::ios::out);
				file.precision(30);

				rapidAssert(file.is_open(), "Unable to open file directory\n");

				file << rows << "\n";
				file << cols << "\n";

				for (unsigned int i = 0; i < rows; i++)
				{
					if (verbose)
					{
						float64 prog = float64(i) / float64(rows);
						for (unsigned int j = 0; j < 20; j++)
						{
							if (j < prog * 20)
								std::cout << "#";
							else
								std::cout << " ";
						}

						auto txt = std::to_string(round(prog * 100., 2));
						txt = txt.substr(0, txt.find_last_not_of("0") + 1);
						std::cout << " | " << txt;

						for (int k = 0; k < 5 - (int) txt.length(); k++)
							std::cout << "0";

						std::cout << "%    \r";
					}

					for (unsigned int j = 0; j < cols; j++)
						file << data[i + j * rows] << " ";
					file << "\n";
				}

				file.close();

				if (verbose)
					std::cout << "#################### | 100.00%\n";
			}

			static Matrix loadFromFile(std::string dir, bool verbose = false)
			{
				std::fstream file;
				file.open(dir, std::ios::in);

				rapidAssert(file.is_open(), "Unable to open directory\n");

				unsigned int rows, cols;

				file >> rows;
				file >> cols;

				auto data = std::vector<dataType>(rows * cols, 0);

				for (unsigned int i = 0; i < rows; i++)
				{
					if (verbose)
					{
						float64 prog = float64(i) / float64(rows);
						for (unsigned int j = 0; j < 20; j++)
						{
							if (j < prog * 20)
								std::cout << "#";
							else
								std::cout << " ";
						}

						auto txt = std::to_string(round(prog * 100., 2));
						txt = txt.substr(0, txt.find_last_not_of("0") + 1);
						std::cout << " | " << txt;

						for (int k = 0; k < 5 - (int) txt.length(); k++)
							std::cout << "0";

						std::cout << "%    \r";
					}

					for (unsigned int j = 0; j < cols; j++)
					{
						file >> data[i + j * rows];

					#ifdef RAPID_CHECK_NAN
						if (data[i + j * rows] != data[i + j * rows])
							message::RapidError("NaN Detected", "NaN detected while loading matrix from file");
					#endif
					}
				}

				auto res = Matrix(rows, cols);
				res.data = data;

				if (verbose)
					std::cout << "#################### | 100.00%\n";

				return res;
			}
		};

	#ifndef RAPID_NO_BLAS
		inline Matrix<float64> Matrix<float64>::dot(const Matrix<float64> &other) const
		{
			rapidAssert(cols == other.rows, "Invalid size for matrix dot product");

			Matrix<float64> res(rows, other.cols);

			size_t M = rows;
			size_t N = cols;
			size_t K = other.cols;

			const float64 *__restrict a = data.data();
			const float64 *__restrict b = other.data.data();
			float64 *__restrict c = res.data.data();

			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1., a, N, b, K, 0., c, K);

		#ifdef RAPID_CHECK_NAN
			res.checkNan("NaN detected in matrix dot product");
		#endif

			return res;
		}

		inline Matrix<float32> Matrix<float32>::dot(const Matrix<float32> &other) const
		{
			rapidAssert(cols == other.rows, "Invalid size for matrix dot product");

			Matrix<float32> res(rows, other.cols);

			size_t M = rows;
			size_t N = cols;
			size_t K = other.cols;

			const float32 *__restrict a = data.data();
			const float32 *__restrict b = other.data.data();
			float32 *__restrict c = res.data.data();

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1., a, N, b, K, 0., c, K);

		#ifdef RAPID_CHECK_NAN
			res.checkNan("NaN detected in matrix dot product");
		#endif

			return res;
		}
	#endif
	}
}
