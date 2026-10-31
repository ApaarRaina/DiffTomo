#pragma once
#include <torch/extension.h>
#include <vector>

torch::Tensor projector_forward(torch::Tensor image, int projections,
                                int detector_bins, float detector_spacing,
                                float coverage, float distance, char type);

torch::Tensor projector_backward(torch::Tensor sinogram,
                                 const std::vector<int64_t> &image_shape,
                                 int projections, int detector_bins,
                                 float detector_spacing, float coverage,
                                 float distance, char type);