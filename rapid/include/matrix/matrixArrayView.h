#pragma once

namespace rapid
{
	namespace matrix
	{
		template<typename T>
		class ArrayView
		{
			T *ptr_;
			uint64 len_;
		public:
			ArrayView(T *ptr, uint64 len) noexcept : ptr_{ptr}, len_{len} {}

			T &operator[](uint64 i) noexcept
			{
				return ptr_[i];
			}

			T &operator[](uint64 i) const noexcept
			{
				return ptr_[i];
			}

			auto size() const noexcept
			{
				return len_;
			}

			auto begin() noexcept
			{
				return ptr_;
			}

			auto end() noexcept
			{
				return ptr_ + len_;
			}
		};
	}
}
