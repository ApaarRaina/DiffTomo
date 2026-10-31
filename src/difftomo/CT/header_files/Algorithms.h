#pragma once // include guard: safe to include multiple times

#include <torch/extension.h>
#include <vector>

torch::Tensor projector_forward_cpu(torch::Tensor image, int projections,
                                    int detector_bins, float detector_spacing,
                                    float coverage, float distance, char type,
                                    int num_threads);

torch::Tensor projector_backward_cpu(torch::Tensor sinogram,
                                     const std::vector<int64_t> &image_shape,
                                     int projections, int detector_bins,
                                     float detector_spacing, float coverage,
                                     float distance, char type,
                                     int num_threads);

torch::Tensor ARTReconstruction(torch::Tensor sinogram,
                                const std::vector<int64_t> &image_shape,
                                int projections, int detector_bins,
                                int iterations, float relaxation_parameter,
                                float detector_spacing, float coverage,
                                float distance, char type);