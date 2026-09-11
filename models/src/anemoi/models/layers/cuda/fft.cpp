// (C) Copyright 2026 Anemoi contributors.
//
// This software is licensed under the terms of the Apache Licence Version 2.0
// which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.

#include <torch/extension.h>

// Implemented in fft.cu.
torch::Tensor rfft_forward(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon
);

torch::Tensor rfft_backward(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon,
    int64_t grid_points
);

torch::Tensor irfft_forward(
    torch::Tensor x,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon,
    int64_t grid_points
);

torch::Tensor irfft_backward(
    torch::Tensor grad_output,
    torch::Tensor offsets,
    torch::Tensor lons,
    int64_t max_nlon,
    int64_t nmodes
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rfft_forward, "Reduced-grid ring RFFT forward");
    m.def("backward", &rfft_backward, "Reduced-grid ring RFFT backward");
    m.def("irfft_forward", &irfft_forward, "Reduced-grid ring IRFFT forward");
    m.def("irfft_backward", &irfft_backward, "Reduced-grid ring IRFFT backward");
}
