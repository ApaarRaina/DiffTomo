#include<stdio.h>
#include<torch/extension.h>


torch::Tensor Forward_line(torch::Tensor image, 
                                torch::Tensor d_volume, 
                                torch::Tensor d_projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_siddon", &Foward_line);
}