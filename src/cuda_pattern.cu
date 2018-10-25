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

#include <mem/cuda_pattern.h>

#define check(ans) (ans)

namespace cuda
{
    template <typename T>
    class device_allocator
        : public std::allocator<T>
    {
    public:
        T* allocate(size_t n)
        {
            void* result = nullptr;

            check(cudaMalloc(&result, n * sizeof(T)));

            return static_cast<T*>(result);
        }

        void deallocate(T* p, size_t n)
        {
            if (p)
            {
                cudaFree(p);
            }
        }
    };

    template <typename T>
    void host_to_device(const T& host_value, T& device_value)
    {
        check(cudaMemcpy(&device_value, &host_value, sizeof(T), cudaMemcpyHostToDevice));
    }

    template <typename T>
    void host_to_device(const T* host_data, T* device_data, size_t length)
    {
        check(cudaMemcpy(device_data, host_data, length * sizeof(T), cudaMemcpyHostToDevice));
    }

    template <typename T>
    void device_to_host(const T& device_value, T& host_value)
    {
        check(cudaMemcpy(&host_value, &device_value, sizeof(T), cudaMemcpyDeviceToHost));
    }

    template <typename T>
    void device_to_host(const T* device_value, T* host_value, size_t length)
    {
        check(cudaMemcpy(host_value, device_value, length * sizeof(T), cudaMemcpyDeviceToHost));
    }
}

namespace mem
{
    namespace internal
    {
        template <typename T>
        __device__ inline bool push(T* values, size_t* size, size_t max_size, T value)
        {
            size_t index = atomicAdd(size, 1);

            if (index < max_size)
            {
                values[index] = value;

                return true;
            }

            return false;
        }

        __global__ void scan_kernel(
            const mem::byte* data,
            size_t data_length,
            const mem::byte* bytes,
            const mem::byte* masks,
            size_t pattern_length,
            size_t* results,
            size_t* results_count,
            size_t max_results)
        {
            size_t total_threads    = blockDim.x * gridDim.x;
            size_t bytes_per_thread = (data_length + total_threads - 1) / total_threads;

            size_t thread_index     = blockIdx.x * blockDim.x + threadIdx.x;
            size_t start_index      = thread_index * bytes_per_thread;
            size_t end_index        = min(start_index + bytes_per_thread, data_length);

            const mem::byte*       current = data + start_index;
            const mem::byte* const end     = data + end_index;

            const size_t last = pattern_length - 1;

            for (; MEM_LIKELY(current < end); ++current)
            {
                size_t i = last;

                do
                {
                    if (MEM_LIKELY((current[i] & masks[i]) != bytes[i]))
                    {
                        goto scan_next;
                    }
                } while (MEM_LIKELY(i--));

                push(results, results_count, max_results, size_t(current - data));

            scan_next:;
            }
        }
    }

    void set_cuda_device(int device)
    {
        check(cudaSetDevice(0));
    }

    device_data::device_data(const byte* host_data, size_t size)
        : device_data_(cuda::device_allocator<byte>{}.allocate(size))
        , size_(size)
    {
        cuda::host_to_device<byte>(host_data, device_data_, size_);
    }

    device_data::~device_data()
    {
        cuda::device_allocator<byte>{}.deallocate(device_data_, size_);
    }

    const byte* device_data::data() const
    {
        return device_data_;
    }

    size_t device_data::size() const
    {
        return size_;
    }

    cuda_pattern::cuda_pattern(const pattern& pattern)
        : size_(pattern.size())
        , trimmed_size_(pattern.trimmed_size())
        , device_bytes_(cuda::device_allocator<byte>{}.allocate(size_))
        , device_masks_(cuda::device_allocator<byte>{}.allocate(size_))
    {
        cuda::host_to_device<byte>(pattern.bytes(), device_bytes_, size_);
        cuda::host_to_device<byte>(pattern.masks(), device_masks_, size_);
    }

    cuda_pattern::~cuda_pattern()
    {
        cuda::device_allocator<byte>{}.deallocate(device_bytes_, size_);
        cuda::device_allocator<byte>{}.deallocate(device_masks_, size_);
    }

    std::vector<size_t> cuda_pattern::scan_all(const device_data& data, size_t max_results)
    {
        size_t scan_length = (data.size() - size_) + 1;

        cudaDeviceProp deviceProp;
        check(cudaGetDeviceProperties(&deviceProp, 0));

        size_t max_threads = deviceProp.maxThreadsPerBlock;
        size_t max_blocks  = 4096;

        size_t thread_count  = min(scan_length, max_threads);
        size_t block_count   = min((scan_length + thread_count - 1) / thread_count, max_blocks);

        size_t* device_results      = cuda::device_allocator<size_t>{}.allocate(max_results);
        size_t* device_result_count = cuda::device_allocator<size_t>{}.allocate(1);

        cuda::host_to_device<size_t>(0, *device_result_count);

        internal::scan_kernel<<<(int) block_count, (int) thread_count>>>(
            data.data(), scan_length,
            device_bytes_, device_masks_, trimmed_size_,
            device_results, device_result_count, max_results);

        size_t result_count = 0;

        cuda::device_to_host<size_t>(*device_result_count, result_count);

        std::vector<size_t> results(min(result_count, max_results));

        cuda::device_to_host<size_t>(device_results, results.data(), results.size());

        return results;
    }
}
