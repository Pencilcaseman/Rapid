#pragma once

#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

#include <array>
#include <string>
#include <vector>
#include <functional>

#include <type_traits>

#include <random>
#include <chrono>
#include <ctime>

#include <algorithm>
#include <iterator>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <regex>
#include <stack>
#include <unordered_map>

#include <conio.h>
#include <sys/stat.h>

#include <intrin.h>

#include <omp.h>

#ifdef _DEBUG
#define RAPID_DEBUG
#else
#define RAPID_RELEASE
#endif

#ifndef RAPID_CUDA
#ifdef CUDA_VERSION
#define RAPID_CUDA
#else
#define RAPID_CPP
#endif
#endif

#ifdef RAPID_CUDA
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>

#pragma comment (lib, "cublas.lib")

// cuBLAS API errors
static const char *HIDDEN_cudaGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR:
			return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "UNKNOWN ERROR";
}

//********************//
// cuBLAS ERROR CHECK //
//********************//
#ifndef cublasSafeCall
#ifdef RAPID_DEBUG
#define cublasSafeCall(err) HIDDEN_cublasSafeCall(err, __FILE__, __LINE__)
#else
#define cublasSafeCall(err) err
#endif
#endif

inline void HIDDEN_cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
	if (CUBLAS_STATUS_SUCCESS != err)
	{
		fprintf(stderr, "cuBLAS error in file '%s', line %d\n \nError: %s \nTERMINATING\n", file, line, HIDDEN_cudaGetErrorEnum(err));
		cudaDeviceReset();
		assert(0);
		exit(1);
	}
}

//********************//
// CUDA ERROR CHECK //
//********************//
#ifndef cudaSafeCall
#ifdef RAPID_DEBUG
#define cudaSafeCall(err) HIDDEN_cudaSafeCall(err, __FILE__, __LINE__)
#else
#define cudaSafeCall(err) err
#endif
#endif

inline void HIDDEN_cudaSafeCall(cudaError_t err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA error in file '%s', line %d\n \nError: %s \nTERMINATING\n", file, line, cudaGetErrorString(err));
		cudaDeviceReset();
		assert(0);
		exit(1);
	}
}
#else
#define cudaSafeCall(func)
#define cublasSafeCall(func)
#endif

#ifdef RAPID_CPP
#ifndef RAPID_NO_AMP
#include <amp.h>		// Optional AMP include
#endif
#else
#ifdef _MSC_VER
#pragma message ("AMP cannot be used in CUDA mode")
#else
#warning AMP cannot be used in CUDA mode
#endif
#endif

#ifndef RAPID_NO_BLAS
#pragma comment(lib, "libopenblas.lib")
#include <cblas.h>		// Optional OpenBLAS include
#endif

#if !(defined(RAPID_NO_AMP) || defined(RAPID_NO_BLAS))
#define RAPID_SET_THREADS(x) omp_set_num_threads((x));
#else
#define RAPID_SET_THREADS(x) (x)
#endif

#undef min
#undef max

// Rapid definitions
#if defined(NDEBUG)
#define RAPID_RELEASE
#else
#ifndef RAPID_DEBUG
#define RAPID_DEBUG
#endif
#endif

#ifdef RAPID_DEBUG
#ifndef RAPID_NO_CHECK_NAN
#define RAPID_CHECK_NAN
#endif
#endif

#if defined(_M_IX86)
#define RAPID_X86
#elif defined(_M_X64)
#define RAPID_X64
#else
#define RAPID_BUILD_UNKNOWN
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#if defined(_WIN32)
#define RAPID_OS_WINDOWS // Windows
#define RAPID_OS "windows"
#elif defined(_WIN64)
#define RAPID_OS_WINDOWS // Windows
#define RAPID_OS "windows"
#elif defined(__CYGWIN__) && !defined(_WIN32)
#define RAPID_OS_WINDOWS  // Windows (Cygwin POSIX under Microsoft Window)
#define RAPID_OS "windows"
#elif defined(__ANDROID__)
#define RAPID_OS_ANDROID // Android (implies Linux, so it must come first)
#define RAPID_OS "android"
#elif defined(__linux__)
#define RAPID_OS_LINUX // Debian, Ubuntu, Gentoo, Fedora, openSUSE, RedHat, Centos and other
#define RAPID_OS "linux"
#elif defined(__unix__) || !defined(__APPLE__) && defined(__MACH__)
#include <sys/param.h>
#if defined(BSD)
#define RAPID_OS_BSD // FreeBSD, NetBSD, OpenBSD, DragonFly BSD
#define RAPID_OS "bsd"
#endif
#elif defined(__hpux)
#define RAPID_OS_HP_UX// HP-UX
#define RAPID_OS "hp-ux"
#elif defined(_AIX)
#define RAPID_OS_AIX // IBM AIX
#define RAPID_OS "aix"
#elif defined(__APPLE__) && defined(__MACH__) // Apple OSX and iOS (Darwin)
#include <TargetConditionals.h>
#if TARGET_IPHONE_SIMULATOR == 1
#define RAPID_OS_IOS // Apple iOS
#define RAPID_OS "ios"
#elif TARGET_OS_IPHONE == 1
#define RAPID_OS_IOS // Apple iOS
#define RAPID_OS "ios"
#elif TARGET_OS_MAC == 1
#define RAPID_OS_OSX // Apple OSX
#define RAPID_OS "osx"
#endif
#elif defined(__sun) && defined(__SVR4)
#define RAPID_OS_SOLARIS // Oracle Solaris, Open Indiana
#define RAPID_OS "solaris"
#else
#define RAPID_OS_UNKNOWN
#define RAPID_OS "unknown"
#endif

#ifdef RAPID_OS_WINDOWS
#include <direct.h>

std::string workingDirectory()
{
	char buff[255];
	char *res = _getcwd(buff, 255);

#ifdef RAPID_DEBUG
	if (res == nullptr)
		throw std::runtime_error("Unable to fetch working directory");
#endif

	std::string current_working_dir(buff);

	return current_working_dir;
}

#define RAPID_WORKING_DIR workingDirectory()
#endif

//**************//
// CURRENT TIME //
//**************//
// A routine to give access to a high precision timer on most systems.
#if defined(_WIN32) || defined(_WIN64) || defined(MSC_VER) || defined(WIN32) || defined(__CYGWIN__) || defined(CUDAAPI)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>

double seconds()
{
	LARGE_INTEGER t;
	static double oofreq;
	static int checkedForHighResTimer = 0;
	static BOOL hasHighResTimer;

	if (!checkedForHighResTimer)
	{
		hasHighResTimer = QueryPerformanceFrequency(&t);
		oofreq = 1.0 / (double) t.QuadPart;
		checkedForHighResTimer = 1;
	}

	if (hasHighResTimer)
	{
		QueryPerformanceCounter(&t);
		return (double) t.QuadPart * oofreq;
	}

	return (double) GetTickCount64() * 1.0e-3;
}

#define TIME (seconds())

#else
#define TIME (omp_get_wtime())
#endif

// Looping for timing things

#define _loop_var rapidTimerLoopIterations
#define _loop_end rapidTimerLoopEnd
#define _loop_timer rapidTimerLoopTimer
#define _loop_goto rapidTimerLoopGoto
#define START_TIMER(id, n) auto _loop_timer##id = rapid::RapidTimer(n);								\
						   for (uint64_t _loop_var##id = 0; _loop_var##id < n; _loop_var##id++) {	\

#define END_TIMER(id)	} \
						_loop_timer##id.endTimer()

namespace rapid
{
	class RapidTimer
	{
	public:
		std::chrono::time_point<std::chrono::steady_clock> start, end;
		uint64_t loops;
		bool finished = false;

		RapidTimer()
		{
			startTimer();
			loops = 1;
		}

		RapidTimer(uint64_t iters)
		{
			startTimer();
			loops = iters;
		}

		~RapidTimer()
		{
			endTimer();
		}

		inline void startTimer()
		{
			start = std::chrono::high_resolution_clock::now();
		}

		inline void endTimer()
		{
			end = std::chrono::high_resolution_clock::now();

			if (finished)
				return;

			finished = true;

			auto delta = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / (double) loops;
			std::string unit = "ns";

			if (delta >= 1000)
			{
				unit = "us";
				delta /= 1000.;
			}

			if (delta >= 1000)
			{
				unit = "ms";
				delta /= 1000.;
			}

			if (delta >= 1000.)
			{
				unit = "s";
				delta /= 1000.;
			}

			std::cout << std::fixed;
			std::cout << "Block finished in: " << delta << " " << unit << "\n";
		}
	};

	template<typename T, typename U>
	inline T rapidCast(const U &in, const std::string &name = "C")
	{
		std::stringstream istr;
		istr.imbue(std::locale(name));

		if (std::is_floating_point<U>::value)
			istr.precision(std::numeric_limits<U>::digits10);

		istr << in;

		std::string str = istr.str(); // save string in case of exception

		T val;
		istr >> val;

		if (istr.fail())
			throw std::invalid_argument(str);

		return val;
	}

	template<>
	inline std::string rapidCast(const std::string &in, const std::string &name)
	{
		return in;
	}

	template<>
	inline unsigned char rapidCast(const std::string &in, const std::string &name)
	{
		return (unsigned char) atoi(in.c_str());
	}

	template<>
	inline char rapidCast(const std::string &in, const std::string &name)
	{
		return (char) atoi(in.c_str());
	}

	template<>
	inline int rapidCast(const std::string &in, const std::string &name)
	{
		return (int) atoi(in.c_str());
	}

	template<>
	inline float rapidCast(const std::string &in, const std::string &name)
	{
		return (float) atof(in.c_str());
	}

	template<>
	inline double rapidCast(const std::string &in, const std::string &name)
	{
		return (double) atof(in.c_str());
	}
}

using uint64 = uint64_t;
using int64 = uint64_t;
using uint32 = uint32_t;
using int32 = uint32_t;
using float32 = float;
using float64 = double;
