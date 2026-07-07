// (C) Copyright 2026 Anemoi contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

#include <torch/extension.h>

// Implemented in ring_fft_cuda.cu.
torch::Tensor ring_rfft_forward_cuda(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon
);

torch::Tensor ring_rfft_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon,
    int64_t grid_points
);

torch::Tensor ring_irfft_forward_cuda(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon,
    int64_t grid_points
);

torch::Tensor ring_irfft_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon,
    int64_t nmodes
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ring_rfft_forward_cuda, "Reduced-grid ring RFFT forward");
    m.def("backward", &ring_rfft_backward_cuda, "Reduced-grid ring RFFT backward");
    m.def("irfft_forward", &ring_irfft_forward_cuda, "Reduced-grid ring IRFFT forward");
    m.def("irfft_backward", &ring_irfft_backward_cuda, "Reduced-grid ring IRFFT backward");
}
