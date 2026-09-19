#include<cuda_runtime.h>
#include<torch/extension.h>


__global__  
    void siddon_kernel(float* image, float* d_volume, float* d_projections, int detector_bins, float detector_spacing, float coverage, float distance) {
    // Kernel implementation for Siddon's algorithm
    // This is a placeholder for the actual implementation
    // You would need to implement the ray tracing and projection logic here
}

torch::Tensor projector_forward(torch::Tensor image, 
                                torch::Tensor d_volume, 
                                torch::Tensor d_projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance)
{
    int blocks = 256; // Number of blocks
    int threads = 256; // Number of threads per block

    forward_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        d_volume.data_ptr<float>(),
        d_projections.data_ptr<float>(),
        detector_bins,
        detector_spacing,
        coverage,
        distance
    );

    return output;
}