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
#include <memory>

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

        void force_init();
        void set_device();

        int get_id() const
        {
            return device_;
        }
    };

    class cuda_device_data
    {
    private:
        class impl;

        std::unique_ptr<impl> impl_;

    public:
        cuda_device_data(cuda_runtime* runtime, const void* data, size_t size);
        ~cuda_device_data();

        cuda_device_data(const cuda_device_data&) = delete;
        cuda_device_data(cuda_device_data&& rhs);

        void wait_for_transfer() const;
         
        const void* data() const;
        size_t size() const;
    };

    class cuda_pattern
    {
    private:
        class impl;

        std::unique_ptr<impl> impl_;

    public:
        explicit cuda_pattern(cuda_runtime* runtime, const pattern& pattern);
        ~cuda_pattern();

        cuda_pattern(const cuda_pattern&) = delete;
        cuda_pattern(cuda_pattern&&);

        std::vector<size_t> scan_all(const cuda_device_data& data, size_t max_results = 1024) const;
    };
}

#endif // MEM_CUDA_PATTERN_BRICK_H
