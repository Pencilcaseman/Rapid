#pragma once

// Pretend this file does not exist.
// I'm warning you...

#include "../internal.h"

namespace rapid
{
	namespace ndarray
	{
		namespace imp
		{
		#define L std::initializer_list
		#define F	if (shape == nullptr) \
					shape = new std::vector<uint64_t>(); \
				shape->emplace_back(data.size()); \
				extractShape<dtype>(*(data.begin()), shape, depth + 1); \
				auto res = std::vector<uint64_t>(shape->begin(), shape->end()); \
				if (depth == 0) delete shape; \
				return res;

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<dtype> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				if (shape == nullptr)
					shape = new std::vector<uint64_t>();
				shape->emplace_back(data.size());

				auto res = std::vector<uint64_t>(shape->begin(), shape->end());
				if (depth == 0) delete shape;
				return res;
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<dtype>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<dtype>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<dtype>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<dtype>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<dtype>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<dtype>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<dtype>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<dtype>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

			template<typename dtype>
			inline std::vector<uint64_t> extractShape(const L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<dtype>>>>>>>>>>>>>>>>>>>> &data, std::vector<uint64_t> *shape = nullptr, uint64_t depth = 0)
			{
				F
			}

		#undef L
		#undef F
		}
	}
}
