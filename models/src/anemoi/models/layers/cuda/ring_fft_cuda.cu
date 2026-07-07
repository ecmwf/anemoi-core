// (C) Copyright 2026 Anemoi contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/complex.h>
#include <torch/extension.h>

#include <cufft.h>

#include <cstdlib>
#include <cstdint>
#include <list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace {

template <typename scalar_t>
__device__ __forceinline__ scalar_t hermitian_mode_factor(const int m, const int nlon) {
    return (m == 0 || (nlon % 2 == 0 && m == nlon / 2)) ? scalar_t(1) : scalar_t(2);
}

// cuFFT works on equal-length batches, so reduced-grid rings are packed by nlon.
template <typename scalar_t>
__global__ void pack_real_rings_kernel(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ ring_indices,
    scalar_t* __restrict__ packed,
    const int64_t lead,
    const int grid_points,
    const int group_count,
    const int nlon
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = lead * group_count * static_cast<int64_t>(nlon);
    if (idx >= total) {
        return;
    }

    const int j = idx % nlon;
    const int group_idx = (idx / nlon) % group_count;
    const int64_t lead_idx = idx / (nlon * group_count);
    const int lat = ring_indices[group_idx];
    packed[idx] = x[lead_idx * grid_points + offsets[lat] + j];
}

template <typename scalar_t>
__global__ void scatter_real_rings_kernel(
    const scalar_t* __restrict__ packed,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ ring_indices,
    scalar_t* __restrict__ output,
    const int64_t lead,
    const int grid_points,
    const int group_count,
    const int nlon,
    const scalar_t scale
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = lead * group_count * static_cast<int64_t>(nlon);
    if (idx >= total) {
        return;
    }

    const int j = idx % nlon;
    const int group_idx = (idx / nlon) % group_count;
    const int64_t lead_idx = idx / (nlon * group_count);
    const int lat = ring_indices[group_idx];
    output[lead_idx * grid_points + offsets[lat] + j] = packed[idx] * scale;
}

template <typename scalar_t>
__global__ void scatter_cufft_rfft_kernel(
    const c10::complex<scalar_t>* __restrict__ packed,
    const int32_t* __restrict__ ring_indices,
    c10::complex<scalar_t>* __restrict__ output,
    const int64_t lead,
    const int nlat,
    const int group_count,
    const int nlon,
    const int nmodes
) {
    const int nfreq = nlon / 2 + 1;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = lead * group_count * static_cast<int64_t>(nmodes);
    if (idx >= total) {
        return;
    }

    const int m = idx % nmodes;
    const int group_idx = (idx / nmodes) % group_count;
    const int64_t lead_idx = idx / (nmodes * group_count);
    const int lat = ring_indices[group_idx];
    const int64_t output_idx = (lead_idx * nlat + lat) * nmodes + m;
    if (m >= nfreq) {
        output[output_idx] = c10::complex<scalar_t>(0, 0);
        return;
    }

    // cuFFT is unnormalised; the Python fallback uses norm="forward".
    const scalar_t scale = scalar_t(1) / scalar_t(nlon);
    const c10::complex<scalar_t> value = packed[(lead_idx * group_count + group_idx) * nfreq + m];
    output[output_idx] = c10::complex<scalar_t>(value.real() * scale, value.imag() * scale);
}

template <typename scalar_t>
__global__ void pack_cufft_rfft_backward_kernel(
    const c10::complex<scalar_t>* __restrict__ grad_output,
    const int32_t* __restrict__ ring_indices,
    c10::complex<scalar_t>* __restrict__ packed,
    const int64_t lead,
    const int nlat,
    const int group_count,
    const int nlon,
    const int nmodes
) {
    const int nfreq = nlon / 2 + 1;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = lead * group_count * static_cast<int64_t>(nfreq);
    if (idx >= total) {
        return;
    }

    const int m = idx % nfreq;
    const int group_idx = (idx / nfreq) % group_count;
    const int64_t lead_idx = idx / (nfreq * group_count);
    if (m >= nmodes) {
        packed[idx] = c10::complex<scalar_t>(0, 0);
        return;
    }

    const int lat = ring_indices[group_idx];
    const c10::complex<scalar_t> grad = grad_output[(lead_idx * nlat + lat) * nmodes + m];
    const bool self_conjugate = m == 0 || (nlon % 2 == 0 && m == nlon / 2);
    // Match torch.fft.rfft backward for the one-sided spectrum.
    const scalar_t scale = self_conjugate ? scalar_t(1) : scalar_t(0.5);
    const scalar_t imag = self_conjugate ? scalar_t(0) : grad.imag() * scale;
    packed[idx] = c10::complex<scalar_t>(grad.real() * scale, imag);
}

template <typename scalar_t>
__global__ void pack_cufft_irfft_forward_kernel(
    const c10::complex<scalar_t>* __restrict__ x,
    const int32_t* __restrict__ ring_indices,
    c10::complex<scalar_t>* __restrict__ packed,
    const int64_t lead,
    const int nlat,
    const int group_count,
    const int nlon,
    const int nmodes
) {
    const int nfreq = nlon / 2 + 1;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = lead * group_count * static_cast<int64_t>(nfreq);
    if (idx >= total) {
        return;
    }

    const int m = idx % nfreq;
    const int group_idx = (idx / nfreq) % group_count;
    const int64_t lead_idx = idx / (nfreq * group_count);
    if (m >= nmodes) {
        packed[idx] = c10::complex<scalar_t>(0, 0);
        return;
    }

    const int lat = ring_indices[group_idx];
    const c10::complex<scalar_t> coeff = x[(lead_idx * nlat + lat) * nmodes + m];
    // cuFFT expects self-conjugate modes to be real.
    if (m == 0 || (nlon % 2 == 0 && m == nlon / 2)) {
        packed[idx] = c10::complex<scalar_t>(coeff.real(), scalar_t(0));
    } else {
        packed[idx] = coeff;
    }
}

template <typename scalar_t>
__global__ void scatter_cufft_irfft_backward_kernel(
    const c10::complex<scalar_t>* __restrict__ packed,
    const int32_t* __restrict__ ring_indices,
    c10::complex<scalar_t>* __restrict__ grad_x,
    const int64_t lead,
    const int nlat,
    const int group_count,
    const int nlon,
    const int nmodes
) {
    const int nfreq = nlon / 2 + 1;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = lead * group_count * static_cast<int64_t>(nmodes);
    if (idx >= total) {
        return;
    }

    const int m = idx % nmodes;
    const int group_idx = (idx / nmodes) % group_count;
    const int64_t lead_idx = idx / (nmodes * group_count);
    const int lat = ring_indices[group_idx];
    const int64_t output_idx = (lead_idx * nlat + lat) * nmodes + m;
    if (m >= nfreq) {
        grad_x[output_idx] = c10::complex<scalar_t>(0, 0);
        return;
    }

    const c10::complex<scalar_t> value = packed[(lead_idx * group_count + group_idx) * nfreq + m];
    const scalar_t scale = hermitian_mode_factor<scalar_t>(m, nlon);
    grad_x[output_idx] = c10::complex<scalar_t>(value.real() * scale, value.imag() * scale);
}

void check_metadata(torch::Tensor offsets, torch::Tensor lons) {
    TORCH_CHECK(offsets.is_cuda(), "offsets must be a CUDA tensor");
    TORCH_CHECK(lons.is_cuda(), "lons must be a CUDA tensor");
    TORCH_CHECK(offsets.is_contiguous(), "offsets must be contiguous");
    TORCH_CHECK(lons.is_contiguous(), "lons must be contiguous");
    TORCH_CHECK(offsets.scalar_type() == at::kInt, "offsets must be int32");
    TORCH_CHECK(lons.scalar_type() == at::kInt, "lons must be int32");
    TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
    TORCH_CHECK(lons.dim() == 1, "lons must be 1D");
    TORCH_CHECK(offsets.size(0) == lons.size(0), "offsets and lons must have the same size");
}

void check_cuda_stream_error() {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int checked_positive_int(const int64_t value, const char* name) {
    TORCH_CHECK(value > 0, name, " must be positive");
    TORCH_CHECK(value <= std::numeric_limits<int>::max(), name, " is too large: ", value);
    return static_cast<int>(value);
}

int checked_nonnegative_int(const int64_t value, const char* name) {
    TORCH_CHECK(value >= 0, name, " must be non-negative");
    TORCH_CHECK(value <= std::numeric_limits<int>::max(), name, " is too large: ", value);
    return static_cast<int>(value);
}

int kernel_blocks(const int64_t total, const int threads) {
    TORCH_CHECK(total >= 0, "CUDA kernel element count must be non-negative");
    TORCH_CHECK(total <= std::numeric_limits<int64_t>::max() - threads + 1, "CUDA kernel element count is too large");
    const int64_t blocks = (total + threads - 1) / threads;
    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "CUDA kernel launch needs too many blocks: ", blocks);
    return static_cast<int>(blocks);
}

void check_cufft(const cufftResult result, const char* message) {
    TORCH_CHECK(result == CUFFT_SUCCESS, message, " (cuFFT error ", static_cast<int>(result), ")");
}

struct CachedCufftPlan {
    cufftHandle handle = 0;
    size_t work_size = 0;
    std::mutex execution_mutex;

    CachedCufftPlan() = default;
    CachedCufftPlan(const CachedCufftPlan&) = delete;
    CachedCufftPlan& operator=(const CachedCufftPlan&) = delete;

    ~CachedCufftPlan() {
        if (handle != 0) {
            cufftDestroy(handle);
        }
    }
};

// cuFFT handles store their stream.
struct CufftPlanKey {
    int device = 0;
    int type = 0;
    int nlon = 0;
    int batch = 0;
    std::uintptr_t stream = 0;

    bool operator<(const CufftPlanKey& other) const {
        return std::tie(device, type, nlon, batch, stream) <
            std::tie(other.device, other.type, other.nlon, other.batch, other.stream);
    }
};

struct CufftPlanCacheEntry {
    std::shared_ptr<CachedCufftPlan> plan;
    std::list<CufftPlanKey>::iterator lru_iterator;
};

std::mutex& cufft_plan_cache_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::map<CufftPlanKey, CufftPlanCacheEntry>& cufft_plan_cache() {
    static auto* cache = new std::map<CufftPlanKey, CufftPlanCacheEntry>();
    return *cache;
}

std::list<CufftPlanKey>& cufft_plan_lru() {
    static auto* lru = new std::list<CufftPlanKey>();
    return *lru;
}

size_t cufft_plan_cache_capacity() {
    static const size_t capacity = []() {
        const char* raw_value = std::getenv("ANEMOI_CUFFT_PLAN_CACHE_SIZE");
        if (raw_value == nullptr || raw_value[0] == '\0') {
            return size_t{2048};
        }

        char* end = nullptr;
        const long parsed = std::strtol(raw_value, &end, 10);
        if (end == raw_value || parsed < 0) {
            return size_t{2048};
        }
        return static_cast<size_t>(parsed);
    }();
    return capacity;
}

std::shared_ptr<CachedCufftPlan> create_cached_cufft_plan(
    const int nlon,
    const int batch,
    const cufftType type,
    const cudaStream_t stream
) {
    auto plan = std::make_shared<CachedCufftPlan>();
    check_cufft(cufftCreate(&plan->handle), "cuFFT handle creation failed");

    int dimensions[] = {nlon};
    const int nfreq = nlon / 2 + 1;
    const bool complex_to_real = type == CUFFT_C2R || type == CUFFT_Z2D;
    const int input_distance = complex_to_real ? nfreq : nlon;
    const int output_distance = complex_to_real ? nlon : nfreq;
    check_cufft(
        cufftMakePlanMany(
            plan->handle,
            1,
            dimensions,
            nullptr,
            1,
            input_distance,
            nullptr,
            1,
            output_distance,
            type,
            batch,
            &plan->work_size
        ),
        "cuFFT plan creation failed"
    );
    check_cufft(cufftSetStream(plan->handle, stream), "cuFFT stream setup failed");
    return plan;
}

std::shared_ptr<CachedCufftPlan> get_cufft_plan(
    const int nlon,
    const int batch,
    const cufftType type,
    const cudaStream_t stream
) {
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    const CufftPlanKey key{
        device,
        static_cast<int>(type),
        nlon,
        batch,
        reinterpret_cast<std::uintptr_t>(stream),
    };

    const size_t capacity = cufft_plan_cache_capacity();
    if (capacity == 0) {
        return create_cached_cufft_plan(nlon, batch, type, stream);
    }

    std::lock_guard<std::mutex> guard(cufft_plan_cache_mutex());
    auto& cache = cufft_plan_cache();
    auto& lru = cufft_plan_lru();

    auto found = cache.find(key);
    if (found != cache.end()) {
        lru.splice(lru.begin(), lru, found->second.lru_iterator);
        return found->second.plan;
    }

    auto plan = create_cached_cufft_plan(nlon, batch, type, stream);
    lru.push_front(key);
    cache.emplace(key, CufftPlanCacheEntry{plan, lru.begin()});

    while (cache.size() > capacity) {
        const CufftPlanKey oldest = lru.back();
        lru.pop_back();
        cache.erase(oldest);
    }

    return plan;
}

struct CufftRingGroup {
    int nlon = 0;
    torch::Tensor ring_indices;
};

struct CufftRingGroups {
    torch::Tensor lons_ref;
    std::vector<CufftRingGroup> groups;
};

// The Python wrapper reuses the lons tensor, so its data pointer is stable.
struct CufftRingGroupsKey {
    int device = 0;
    int64_t nlat = 0;
    std::uintptr_t lons_data = 0;

    bool operator<(const CufftRingGroupsKey& other) const {
        return std::tie(device, nlat, lons_data) < std::tie(other.device, other.nlat, other.lons_data);
    }
};

std::mutex& cufft_ring_groups_cache_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::map<CufftRingGroupsKey, std::shared_ptr<const CufftRingGroups>>& cufft_ring_groups_cache() {
    static auto* cache = new std::map<CufftRingGroupsKey, std::shared_ptr<const CufftRingGroups>>();
    return *cache;
}

std::map<int, std::vector<int32_t>> make_nlon_groups(torch::Tensor lons) {
    auto lons_cpu = lons.to(torch::kCPU);
    const auto* lons_ptr = lons_cpu.data_ptr<int32_t>();

    std::map<int, std::vector<int32_t>> groups;
    for (int32_t lat = 0; lat < lons_cpu.size(0); ++lat) {
        groups[static_cast<int>(lons_ptr[lat])].push_back(lat);
    }
    return groups;
}

torch::Tensor ring_indices_to_cuda(const std::vector<int32_t>& ring_indices, const torch::Device device) {
    auto ring_indices_cpu = torch::empty(
        {static_cast<int64_t>(ring_indices.size())},
        torch::TensorOptions().dtype(at::kInt).device(torch::kCPU)
    );
    auto* ring_indices_ptr = ring_indices_cpu.data_ptr<int32_t>();
    for (size_t i = 0; i < ring_indices.size(); ++i) {
        ring_indices_ptr[i] = ring_indices[i];
    }
    return ring_indices_cpu.to(device);
}

std::shared_ptr<const CufftRingGroups> get_cufft_ring_groups(torch::Tensor lons) {
    const CufftRingGroupsKey key{
        lons.get_device(),
        lons.size(0),
        reinterpret_cast<std::uintptr_t>(lons.data_ptr<int32_t>()),
    };

    std::lock_guard<std::mutex> guard(cufft_ring_groups_cache_mutex());
    auto& cache = cufft_ring_groups_cache();
    auto found = cache.find(key);
    if (found != cache.end()) {
        return found->second;
    }

    auto groups = std::make_shared<CufftRingGroups>();
    groups->lons_ref = lons;
    for (const auto& [nlon, ring_indices_host] : make_nlon_groups(lons)) {
        groups->groups.push_back(CufftRingGroup{
            nlon,
            ring_indices_to_cuda(ring_indices_host, lons.device()),
        });
    }
    cache.emplace(key, groups);
    return groups;
}

int checked_cufft_batch(const int64_t batch) {
    TORCH_CHECK(batch <= std::numeric_limits<int>::max(), "cuFFT batch is too large: ", batch);
    return static_cast<int>(batch);
}

template <typename scalar_t>
cufftType cufft_r2c_type();

template <>
cufftType cufft_r2c_type<float>() {
    return CUFFT_R2C;
}

template <>
cufftType cufft_r2c_type<double>() {
    return CUFFT_D2Z;
}

template <typename scalar_t>
cufftType cufft_c2r_type();

template <>
cufftType cufft_c2r_type<float>() {
    return CUFFT_C2R;
}

template <>
cufftType cufft_c2r_type<double>() {
    return CUFFT_Z2D;
}

template <typename scalar_t>
void cufft_exec_r2c(cufftHandle plan, scalar_t* input, c10::complex<scalar_t>* output);

template <>
void cufft_exec_r2c<float>(cufftHandle plan, float* input, c10::complex<float>* output) {
    check_cufft(
        cufftExecR2C(plan, reinterpret_cast<cufftReal*>(input), reinterpret_cast<cufftComplex*>(output)),
        "cuFFT R2C execution failed"
    );
}

template <>
void cufft_exec_r2c<double>(cufftHandle plan, double* input, c10::complex<double>* output) {
    check_cufft(
        cufftExecD2Z(plan, reinterpret_cast<cufftDoubleReal*>(input), reinterpret_cast<cufftDoubleComplex*>(output)),
        "cuFFT D2Z execution failed"
    );
}

template <typename scalar_t>
void cufft_exec_c2r(cufftHandle plan, c10::complex<scalar_t>* input, scalar_t* output);

template <>
void cufft_exec_c2r<float>(cufftHandle plan, c10::complex<float>* input, float* output) {
    check_cufft(
        cufftExecC2R(plan, reinterpret_cast<cufftComplex*>(input), reinterpret_cast<cufftReal*>(output)),
        "cuFFT C2R execution failed"
    );
}

template <>
void cufft_exec_c2r<double>(cufftHandle plan, c10::complex<double>* input, double* output) {
    check_cufft(
        cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(input), reinterpret_cast<cufftDoubleReal*>(output)),
        "cuFFT Z2D execution failed"
    );
}

// Each cuFFT path works group by group: pack, transform, scatter.
template <typename scalar_t>
void launch_cufft_rfft_forward(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    torch::Tensor output,
    const cudaStream_t stream,
    const int64_t lead,
    const int grid_points,
    const int nlat,
    const int nmodes
) {
    constexpr int threads = 256;
    const auto ring_groups = get_cufft_ring_groups(lons);
    for (const auto& group : ring_groups->groups) {
        const int nlon = group.nlon;
        const torch::Tensor& ring_indices = group.ring_indices;
        const int group_count = static_cast<int>(ring_indices.size(0));
        const int nfreq = nlon / 2 + 1;
        const int64_t batch64 = lead * static_cast<int64_t>(group_count);
        const int batch = checked_cufft_batch(batch64);
        auto packed = torch::empty({batch64, nlon}, x.options());
        auto cufft_output = torch::empty({batch64, nfreq}, output.options());

        pack_real_rings_kernel<scalar_t><<<
            kernel_blocks(batch64 * nlon, threads),
            threads,
            0,
            stream>>>(
            x.data_ptr<scalar_t>(),
            offsets.data_ptr<int32_t>(),
            ring_indices.data_ptr<int32_t>(),
            packed.data_ptr<scalar_t>(),
            lead,
            grid_points,
            group_count,
            nlon
        );
        check_cuda_stream_error();

        auto plan = get_cufft_plan(nlon, batch, cufft_r2c_type<scalar_t>(), stream);
        {
            std::lock_guard<std::mutex> guard(plan->execution_mutex);
            cufft_exec_r2c(plan->handle, packed.data_ptr<scalar_t>(), cufft_output.data_ptr<c10::complex<scalar_t>>());
        }

        scatter_cufft_rfft_kernel<scalar_t><<<
            kernel_blocks(batch64 * nmodes, threads),
            threads,
            0,
            stream>>>(
            cufft_output.data_ptr<c10::complex<scalar_t>>(),
            ring_indices.data_ptr<int32_t>(),
            output.data_ptr<c10::complex<scalar_t>>(),
            lead,
            nlat,
            group_count,
            nlon,
            nmodes
        );
        check_cuda_stream_error();
    }
}

template <typename scalar_t>
void launch_cufft_rfft_backward(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    torch::Tensor grad_x,
    const cudaStream_t stream,
    const int64_t lead,
    const int grid_points,
    const int nlat,
    const int nmodes
) {
    constexpr int threads = 256;
    const auto ring_groups = get_cufft_ring_groups(lons);
    for (const auto& group : ring_groups->groups) {
        const int nlon = group.nlon;
        const torch::Tensor& ring_indices = group.ring_indices;
        const int group_count = static_cast<int>(ring_indices.size(0));
        const int nfreq = nlon / 2 + 1;
        const int64_t batch64 = lead * static_cast<int64_t>(group_count);
        const int batch = checked_cufft_batch(batch64);
        auto packed_grad = torch::empty({batch64, nfreq}, grad_output.options());
        auto packed_real = torch::empty({batch64, nlon}, grad_x.options());

        pack_cufft_rfft_backward_kernel<scalar_t><<<
            kernel_blocks(batch64 * nfreq, threads),
            threads,
            0,
            stream>>>(
            grad_output.data_ptr<c10::complex<scalar_t>>(),
            ring_indices.data_ptr<int32_t>(),
            packed_grad.data_ptr<c10::complex<scalar_t>>(),
            lead,
            nlat,
            group_count,
            nlon,
            nmodes
        );
        check_cuda_stream_error();

        auto plan = get_cufft_plan(nlon, batch, cufft_c2r_type<scalar_t>(), stream);
        {
            std::lock_guard<std::mutex> guard(plan->execution_mutex);
            cufft_exec_c2r(plan->handle, packed_grad.data_ptr<c10::complex<scalar_t>>(), packed_real.data_ptr<scalar_t>());
        }

        scatter_real_rings_kernel<scalar_t><<<
            kernel_blocks(batch64 * nlon, threads),
            threads,
            0,
            stream>>>(
            packed_real.data_ptr<scalar_t>(),
            offsets.data_ptr<int32_t>(),
            ring_indices.data_ptr<int32_t>(),
            grad_x.data_ptr<scalar_t>(),
            lead,
            grid_points,
            group_count,
            nlon,
            scalar_t(1) / scalar_t(nlon)
        );
        check_cuda_stream_error();
    }
}

template <typename scalar_t>
void launch_cufft_irfft_forward(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    torch::Tensor output,
    const cudaStream_t stream,
    const int64_t lead,
    const int grid_points,
    const int nlat,
    const int nmodes
) {
    constexpr int threads = 256;
    const auto ring_groups = get_cufft_ring_groups(lons);
    for (const auto& group : ring_groups->groups) {
        const int nlon = group.nlon;
        const torch::Tensor& ring_indices = group.ring_indices;
        const int group_count = static_cast<int>(ring_indices.size(0));
        const int nfreq = nlon / 2 + 1;
        const int64_t batch64 = lead * static_cast<int64_t>(group_count);
        const int batch = checked_cufft_batch(batch64);
        auto packed_coeffs = torch::empty({batch64, nfreq}, x.options());
        auto packed_real = torch::empty({batch64, nlon}, output.options());

        pack_cufft_irfft_forward_kernel<scalar_t><<<
            kernel_blocks(batch64 * nfreq, threads),
            threads,
            0,
            stream>>>(
            x.data_ptr<c10::complex<scalar_t>>(),
            ring_indices.data_ptr<int32_t>(),
            packed_coeffs.data_ptr<c10::complex<scalar_t>>(),
            lead,
            nlat,
            group_count,
            nlon,
            nmodes
        );
        check_cuda_stream_error();

        auto plan = get_cufft_plan(nlon, batch, cufft_c2r_type<scalar_t>(), stream);
        {
            std::lock_guard<std::mutex> guard(plan->execution_mutex);
            cufft_exec_c2r(plan->handle, packed_coeffs.data_ptr<c10::complex<scalar_t>>(), packed_real.data_ptr<scalar_t>());
        }

        scatter_real_rings_kernel<scalar_t><<<
            kernel_blocks(batch64 * nlon, threads),
            threads,
            0,
            stream>>>(
            packed_real.data_ptr<scalar_t>(),
            offsets.data_ptr<int32_t>(),
            ring_indices.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(),
            lead,
            grid_points,
            group_count,
            nlon,
            scalar_t(1)
        );
        check_cuda_stream_error();
    }
}

template <typename scalar_t>
void launch_cufft_irfft_backward(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    torch::Tensor grad_x,
    const cudaStream_t stream,
    const int64_t lead,
    const int grid_points,
    const int nlat,
    const int nmodes
) {
    constexpr int threads = 256;
    const auto ring_groups = get_cufft_ring_groups(lons);
    for (const auto& group : ring_groups->groups) {
        const int nlon = group.nlon;
        const torch::Tensor& ring_indices = group.ring_indices;
        const int group_count = static_cast<int>(ring_indices.size(0));
        const int nfreq = nlon / 2 + 1;
        const int64_t batch64 = lead * static_cast<int64_t>(group_count);
        const int batch = checked_cufft_batch(batch64);
        auto packed_real = torch::empty({batch64, nlon}, grad_output.options());
        auto packed_coeffs = torch::empty({batch64, nfreq}, grad_x.options());

        pack_real_rings_kernel<scalar_t><<<
            kernel_blocks(batch64 * nlon, threads),
            threads,
            0,
            stream>>>(
            grad_output.data_ptr<scalar_t>(),
            offsets.data_ptr<int32_t>(),
            ring_indices.data_ptr<int32_t>(),
            packed_real.data_ptr<scalar_t>(),
            lead,
            grid_points,
            group_count,
            nlon
        );
        check_cuda_stream_error();

        auto plan = get_cufft_plan(nlon, batch, cufft_r2c_type<scalar_t>(), stream);
        {
            std::lock_guard<std::mutex> guard(plan->execution_mutex);
            cufft_exec_r2c(plan->handle, packed_real.data_ptr<scalar_t>(), packed_coeffs.data_ptr<c10::complex<scalar_t>>());
        }

        scatter_cufft_irfft_backward_kernel<scalar_t><<<
            kernel_blocks(batch64 * nmodes, threads),
            threads,
            0,
            stream>>>(
            packed_coeffs.data_ptr<c10::complex<scalar_t>>(),
            ring_indices.data_ptr<int32_t>(),
            grad_x.data_ptr<c10::complex<scalar_t>>(),
            lead,
            nlat,
            group_count,
            nlon,
            nmodes
        );
        check_cuda_stream_error();
    }
}

}  // namespace

// Python calls these four functions.
torch::Tensor ring_rfft_forward_cuda(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    const int64_t max_nlon
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must have shape [lead, grid]");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble, "x must be float32 or float64");
    check_metadata(offsets, lons);

    const int max_nlon_int = checked_positive_int(max_nlon, "max_nlon");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    const int64_t lead = x.size(0);
    const int grid_points = checked_positive_int(x.size(1), "x grid dimension");
    const int nlat = checked_positive_int(lons.size(0), "latitude count");
    const int nmodes = max_nlon_int / 2 + 1;
    const auto complex_dtype = x.scalar_type() == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    auto output = torch::empty({lead, nlat, nmodes}, x.options().dtype(complex_dtype));

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "ring_rfft_cufft_forward_cuda", [&] {
        launch_cufft_rfft_forward<scalar_t>(
            x,
            offsets,
            lons,
            output,
            stream,
            lead,
            grid_points,
            nlat,
            nmodes
        );
    });
    return output;
}

torch::Tensor ring_rfft_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    const int64_t max_nlon,
    const int64_t grid_points
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output must have shape [lead, nlat, modes]");
    TORCH_CHECK(
        grad_output.scalar_type() == at::kComplexFloat || grad_output.scalar_type() == at::kComplexDouble,
        "grad_output must be complex64 or complex128"
    );
    check_metadata(offsets, lons);

    const int max_nlon_int = checked_positive_int(max_nlon, "max_nlon");
    const int grid_points_int = checked_positive_int(grid_points, "grid_points");

    const int nlat = checked_positive_int(lons.size(0), "latitude count");
    const int nmodes = max_nlon_int / 2 + 1;
    TORCH_CHECK(grad_output.size(1) == nlat, "grad_output latitude dimension must match metadata");
    TORCH_CHECK(grad_output.size(2) == nmodes, "grad_output mode dimension must be max_nlon / 2 + 1");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    const int64_t lead = grad_output.size(0);
    const auto real_dtype = grad_output.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
    auto grad_x = torch::empty({lead, grid_points}, grad_output.options().dtype(real_dtype));

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(real_dtype, "ring_rfft_cufft_backward_cuda", [&] {
        launch_cufft_rfft_backward<scalar_t>(
            grad_output,
            offsets,
            lons,
            grad_x,
            stream,
            lead,
            grid_points_int,
            nlat,
            nmodes
        );
    });
    return grad_x;
}

torch::Tensor ring_irfft_forward_cuda(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    const int64_t max_nlon,
    const int64_t grid_points
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must have shape [lead, nlat, modes]");
    TORCH_CHECK(
        x.scalar_type() == at::kComplexFloat || x.scalar_type() == at::kComplexDouble,
        "x must be complex64 or complex128"
    );
    check_metadata(offsets, lons);

    const int max_nlon_int = checked_positive_int(max_nlon, "max_nlon");
    const int grid_points_int = checked_positive_int(grid_points, "grid_points");

    const int nlat = checked_positive_int(lons.size(0), "latitude count");
    const int nmodes = checked_positive_int(x.size(2), "x mode dimension");
    TORCH_CHECK(x.size(1) == nlat, "x latitude dimension must match metadata");
    TORCH_CHECK(nmodes <= max_nlon_int / 2 + 1, "x mode dimension must be <= max_nlon / 2 + 1");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    const int64_t lead = x.size(0);
    const auto real_dtype = x.scalar_type() == at::kComplexFloat ? at::kFloat : at::kDouble;
    auto output = torch::empty({lead, grid_points}, x.options().dtype(real_dtype));

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(real_dtype, "ring_irfft_cufft_forward_cuda", [&] {
        launch_cufft_irfft_forward<scalar_t>(
            x,
            offsets,
            lons,
            output,
            stream,
            lead,
            grid_points_int,
            nlat,
            nmodes
        );
    });
    return output;
}

torch::Tensor ring_irfft_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    const int64_t max_nlon,
    const int64_t nmodes
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.dim() == 2, "grad_output must have shape [lead, grid]");
    TORCH_CHECK(
        grad_output.scalar_type() == at::kFloat || grad_output.scalar_type() == at::kDouble,
        "grad_output must be float32 or float64"
    );
    check_metadata(offsets, lons);

    const int max_nlon_int = checked_positive_int(max_nlon, "max_nlon");
    const int nmodes_int = checked_positive_int(nmodes, "nmodes");
    TORCH_CHECK(nmodes_int <= max_nlon_int / 2 + 1, "nmodes must be <= max_nlon / 2 + 1");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_output));
    const int64_t lead = grad_output.size(0);
    const int grid_points = checked_positive_int(grad_output.size(1), "grad_output grid dimension");
    const int nlat = checked_positive_int(lons.size(0), "latitude count");
    const auto complex_dtype = grad_output.scalar_type() == at::kFloat ? at::kComplexFloat : at::kComplexDouble;
    auto grad_x = torch::empty({lead, nlat, nmodes_int}, grad_output.options().dtype(complex_dtype));

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "ring_irfft_cufft_backward_cuda", [&] {
        launch_cufft_irfft_backward<scalar_t>(
            grad_output,
            offsets,
            lons,
            grad_x,
            stream,
            lead,
            grid_points,
            nlat,
            nmodes_int
        );
    });
    return grad_x;
}
