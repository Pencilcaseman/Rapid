#pragma once

namespace rapid
{
	namespace matrix
	{
		template<typename T>
		class ArrayView
		{
			T *ptr_;
			size_t len_;
		public:
			ArrayView(T *ptr, size_t len) noexcept : ptr_{ptr}, len_{len} {}

			T &operator[](size_t i) noexcept
			{
				return ptr_[i];
			}

			T &operator[](size_t i) const noexcept
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
