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

#include "mem/cuda_pattern.h"

#include <stdexcept>

#define check(ans) do { assert_((ans), __FILE__, __LINE__); } while (false)
inline void assert_(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        char buffer[1024];

        snprintf(buffer, 1024, "CUDA Check Failed: %s : %s : %i", cudaGetErrorString(code), file, line);

        throw std::runtime_error(buffer);
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
            const byte* data,
            size_t data_length,
            const byte* bytes,
            const byte* masks,
            size_t pattern_length,
            size_t* results,
            size_t* results_count,
            size_t max_results)
        {
            size_t thread_index  = blockIdx.x * blockDim.x + threadIdx.x;
            size_t total_threads = blockDim.x * gridDim.x;

            size_t bytes_per_thread = (data_length + total_threads - 1) / total_threads;

            size_t start_index      = thread_index * bytes_per_thread;
            size_t end_index        = min(start_index + bytes_per_thread, data_length);

            const byte*       current = data + start_index;
            const byte* const end     = data + end_index;

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

    class cuda_device_data::impl
    {
    private:
        void* data_ {nullptr};
        size_t size_ {0};
        cudaStream_t stream_;

    public:
        impl::impl(cuda_runtime* runtime, const void* data, size_t size)
        {
            runtime->set_device();

            check(cudaStreamCreate(&stream_));
            check(cudaMalloc(&data_, size));
            check(cudaMemcpyAsync(data_, data, size, cudaMemcpyHostToDevice, stream_));

            size_ = size;
        }

        impl::~impl()
        {
            check(cudaFree(data_));
            check(cudaStreamDestroy(stream_));
        }

        void wait_for_transfer() const
        {
            check(cudaStreamSynchronize(stream_));
        }

        const void* data() const
        {
            return data_;
        }

        size_t size() const
        {
            return size_;
        }
    };

    cuda_device_data::cuda_device_data(cuda_runtime* runtime, const void* data, size_t size)
        : impl_(new impl(runtime, data, size))
    { }

    cuda_device_data::~cuda_device_data() = default;
    cuda_device_data::cuda_device_data(cuda_device_data&& rhs) = default;

    void cuda_device_data::wait_for_transfer() const
    {
        return impl_->wait_for_transfer();
    }

    const void* cuda_device_data::data() const
    {
        return impl_->data();
    }

    size_t cuda_device_data::size() const
    {
        return impl_->size();
    }

    cuda_runtime::cuda_runtime(int device)
        : device_(device)
    { }

    cuda_runtime::~cuda_runtime() = default;

    void cuda_runtime::force_init()
    {
        set_device();

        check(cudaFree(nullptr));
    }

    void cuda_runtime::set_device()
    {
        check(cudaSetDevice(device_));
    }

    class cuda_pattern::impl
    {
    private:
        cuda_runtime* runtime_ {nullptr};
        void* bytes_ {nullptr};
        void* masks_ {nullptr};
        size_t size_ {0};
        size_t trimmed_size_  {0};
        cudaStream_t stream_;

    public:
        impl(cuda_runtime* runtime, const pattern& pattern)
            : runtime_(runtime)
        {
            runtime_->set_device();

            check(cudaStreamCreate(&stream_));

            size_t size = pattern.size();

            check(cudaMalloc(&bytes_, size));
            check(cudaMalloc(&masks_, size));

            check(cudaMemcpyAsync(bytes_, pattern.bytes(), size, cudaMemcpyHostToDevice, stream_));
            check(cudaMemcpyAsync(masks_, pattern.masks(), size, cudaMemcpyHostToDevice, stream_));

            size_ = size;
            trimmed_size_ = pattern.trimmed_size();
        }

        ~impl()
        {
            check(cudaFree(bytes_));
            check(cudaFree(masks_));

            check(cudaStreamDestroy(stream_));
        }

        void wait_for_transfer() const
        {
            check(cudaStreamSynchronize(stream_));
        }

        std::vector<size_t> scan_all(const cuda_device_data& data, size_t max_results) const
        {
            runtime_->set_device();

            if ((data.size() < size_) || (trimmed_size_ == 0))
            {
                return {};
            }

            size_t scan_length = (data.size() - size_) + 1;

            cudaDeviceProp deviceProp;
            check(cudaGetDeviceProperties(&deviceProp, runtime_->get_id()));

            size_t max_threads = deviceProp.maxThreadsPerBlock;
            size_t max_blocks  = 1024;

            size_t thread_count  = min(scan_length, max_threads);
            size_t block_count   = min((scan_length + thread_count - 1) / thread_count, max_blocks);

            size_t* device_results      = nullptr;
            size_t* device_result_count = nullptr;

            check(cudaMalloc((void**) &device_results, max_results * sizeof(size_t)));
            check(cudaMalloc((void**) &device_result_count, sizeof(size_t)));

            const size_t zero = 0;

            cudaStream_t stream;

            check(cudaStreamCreate(&stream));

            check(cudaMemcpyAsync(device_result_count, &zero, sizeof(size_t), cudaMemcpyHostToDevice, stream));

            wait_for_transfer();
            data.wait_for_transfer();

            internal::scan_kernel<<<(int) block_count, (int) thread_count, 0, stream>>>(
                (const byte*) data.data(), scan_length,
                (const byte*) bytes_, (const byte*) masks_, trimmed_size_,
                device_results, device_result_count, max_results);

            size_t result_count = 0;

            check(cudaMemcpyAsync(&result_count, device_result_count, sizeof(size_t), cudaMemcpyDeviceToHost, stream));

            std::vector<size_t> results(min(result_count, max_results));

            check(cudaMemcpyAsync(results.data(), device_results, results.size() * sizeof(size_t), cudaMemcpyDeviceToHost, stream));

            check(cudaStreamDestroy(stream));

            check(cudaFree(device_results));
            check(cudaFree(device_result_count));

            return results;
        }
    };

    cuda_pattern::cuda_pattern(cuda_runtime* runtime, const pattern& pattern)
        : impl_(new impl(runtime, pattern))
    { }

    cuda_pattern::~cuda_pattern() = default;
    cuda_pattern::cuda_pattern(cuda_pattern&&) = default;

    std::vector<size_t> cuda_pattern::scan_all(const cuda_device_data& data, size_t max_results) const
    {
        return impl_->scan_all(data, max_results);
    }
}
