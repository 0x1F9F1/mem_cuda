/*
    Copyright 2018 Brick

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
    BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#if !defined(MEM_CUDA_PATTERN_BRICK_H)
#define MEM_CUDA_PATTERN_BRICK_H

#include <mem/pattern.h>
#include <vector>

namespace mem
{
    class cuda_runtime
    {
    private:
        int device_;

    public:
        explicit cuda_runtime(int device = 0);
        ~cuda_runtime();

        cuda_runtime(const cuda_runtime&) = delete;
        cuda_runtime(cuda_runtime&&) = delete;

        void set_device();
    };

    class cuda_device_data
    {
    private:
        void* data_ {nullptr};
        size_t size_ {0};

    public:
        cuda_device_data(cuda_runtime* runtime, const void* data, size_t size);
        ~cuda_device_data();

        cuda_device_data(const cuda_device_data&) = delete;
        cuda_device_data(cuda_device_data&& rhs);

        const void* data() const
        {
            return data_;
        }

        size_t size() const
        {
            return size_;
        }
    };

    class cuda_pattern
    {
    private:
        void* bytes_ {nullptr};
        void* masks_ {nullptr};
        size_t size_ {0};
        size_t trimmed_size_ {0};

    public:
        explicit cuda_pattern(cuda_runtime* runtime, const pattern& pattern);
        ~cuda_pattern();

        cuda_pattern(const cuda_pattern&) = delete;
        cuda_pattern(cuda_pattern&&) = delete;

        std::vector<size_t> scan_all(const cuda_device_data& data, size_t max_results = 1024) const;
    };
}

#endif // MEM_CUDA_PATTERN_BRICK_H
