#include<stdio.h>
#include<torch/extension.h>


torch::Tensor projector_forward(torch::Tensor image,  
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance);

torch::Tensor projector_backward(torch::Tensor sinogram,
                                const std::vector<int64_t>& image_shape, 
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_siddon", &projector_forward);
    m.def("backward_siddon", &projector_backward);
}