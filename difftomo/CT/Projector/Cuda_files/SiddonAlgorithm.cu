#include<cuda_runtime.h>
#include<torch/extension.h>

__global__  
    void siddon_kernel(float* image, float* output, int batches, int channels, int height, int width, int projections, int detector_bins, float detector_spacing, float coverage, float distance, char type) {

        int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;

        int blocksPerProjection =
            (detector_bins + threadsPerBlock - 1) / threadsPerBlock;

        int blocksPerImage = projections * blocksPerProjection;

        int image_index = blockIdx.x / blocksPerImage;
        int block_in_image = blockIdx.x % blocksPerImage;

        int batch_index = image_index / channels;
        int channel_index = image_index % channels;

        int projection_index = block_in_image / blocksPerProjection;

        int block_in_projection = block_in_image % blocksPerProjection;

        int detector_index =block_in_projection * threadsPerBlock + threadIdx.x;

        // stopping illegal threads from executing
        if (projection_index >= projections || detector_index >= detector_bins) 
            return;

        float bx = -(height / 2.0f); 
        float by = -(width / 2.0f);
        float u = detector_spacing * (detector_index - detector_bins / 2.0f);

        // thread starts executing here so calculate ray for this detector index and projection index
        float p1[2];
        float p2[2];
        float angle;

        if (type == 'p'){
            angle = projection_index * coverage / projections;

            p1[0] = distance* cosf(angle) - u * sinf(angle);
            p1[1] = distance* sinf(angle) + u * cosf(angle);

            p2[0] = -distance * cosf(angle) - u * sinf(angle);
            p2[1] = -distance * sinf(angle) + u * cosf(angle);
        }

        else if (type == 'f'){
            angle = projection_index * coverage / projections;

            p1[0] = distance* sinf(angle);
            p1[1] = distance * cosf(angle);

            p2[0] = -distance * sinf(angle) + u * cosf(angle);
            p2[1] = -distance * cosf(angle) - u * sinf(angle);
        }
        
        float dx = p2[0] - p1[0];
        float dy = p2[1] - p1[1];
        float d_conv = sqrtf(dx * dx + dy * dy);

        // Parametric boundaries for x and y planes
        float ax_min = (std::abs(dx) > 1e-6f) ? min((bx - p1[0]) / dx, (bx + height - p1[0]) / dx) : -1e9f;
        float ax_max = (std::abs(dx) > 1e-6f) ? max((bx - p1[0]) / dx, (bx + height - p1[0]) / dx) :  1e9f;

        float ay_min = (std::abs(dy) > 1e-6f) ? min((by - p1[1]) / dy, (by + width - p1[1]) / dy) : -1e9f;
        float ay_max = (std::abs(dy) > 1e-6f) ? max((by - p1[1]) / dy, (by + width - p1[1]) / dy) :  1e9f;

        float alpha_min = max(0.0f, max(ax_min, ay_min));
        float alpha_max = min(1.0f, min(ax_max, ay_max));
        
        float d_ax = (std::abs(dx) > 1e-6f) ? 1.0f / std::abs(dx) : 1e9f;
        float d_ay = (std::abs(dy) > 1e-6f) ? 1.0f / std::abs(dy) : 1e9f;

        // Initial alpha values
        float ax_next = 1e9f;
        if (std::abs(dx) > 1e-6f) {
            float rx_entry = p1[0] + alpha_min * dx - bx;
            int next_i = (dx > 0) ? (int)floorf(rx_entry) + 1 : (int)floorf(rx_entry);
            ax_next = (bx + (float)next_i - p1[0]) / dx;
            if (ax_next <= alpha_min) ax_next += d_ax;
        }

        float ay_next = 1e9f;
        if (std::abs(dy) > 1e-6f) {
            float ry_entry = p1[1] + alpha_min * dy - by;
            int next_j = (dy > 0) ? (int)floorf(ry_entry) + 1 : (int)floorf(ry_entry);
            ay_next = (by + (float)next_j - p1[1]) / dy;
            if (ay_next <= alpha_min) ay_next += d_ay;
        }
        

        float alpha_curr = alpha_min;
        float sum = 0.0f;
        
        // Ray marching loop
        while (alpha_curr < alpha_max - 1e-6f) {
            float alpha_next = min(alpha_max, min(ax_next, ay_next));
            float mid_alpha = (alpha_curr + alpha_next) * 0.5f;

            int img_i = (int)floorf(p1[0] + mid_alpha * dx - bx);
            int img_j = (int)floorf(p1[1] + mid_alpha * dy - by);

            if (img_i >= 0 && img_i < height && img_j >= 0 && img_j < width) {
                float length = (alpha_next - alpha_curr) * d_conv;
                sum += length * image[batch_index * channels * height * width + channel_index * height * width + img_i * width + img_j];
            }

            if (ax_next < ay_next) {
                ax_next += d_ax;
            } else {
                ay_next += d_ay;
            }
            alpha_curr = alpha_next;
        }

        output[batch_index * channels * projections * detector_bins + channel_index * projections * detector_bins + projection_index * detector_bins + detector_index] = sum;
        
    }

__global__ void backprojection_kernel(float* sinogram, float* output, int batches, int channels, int height, int width, int projections, int detector_bins, float detector_spacing, float coverage, float distance, char type) {
        int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;

        int blocksPerProjection =
            (detector_bins + threadsPerBlock - 1) / threadsPerBlock;

        int blocksPerImage = projections * blocksPerProjection;

        int image_index = blockIdx.x / blocksPerImage;
        int block_in_image = blockIdx.x % blocksPerImage;

        int batch_index = image_index / channels;
        int channel_index = image_index % channels;

        int projection_index = block_in_image / blocksPerProjection;

        int block_in_projection = block_in_image % blocksPerProjection;

        int detector_index =block_in_projection * threadsPerBlock + threadIdx.x;

        // stopping illegal threads from executing
        if (projection_index >= projections || detector_index >= detector_bins) 
            return;

        float bx = -(height / 2.0f); 
        float by = -(width / 2.0f);
        float u = detector_spacing * (detector_index - detector_bins / 2.0f);

        // thread starts executing here so calculate ray for this detector index and projection index
        float p1[2];
        float p2[2];
        float angle;

        if (type == 'p'){
            angle = projection_index * coverage / projections;

            p1[0] = distance* cosf(angle) - u * sinf(angle);
            p1[1] = distance* sinf(angle) + u * cosf(angle);

            p2[0] = -distance * cosf(angle) - u * sinf(angle);
            p2[1] = -distance * sinf(angle) + u * cosf(angle);
        }

        else if (type == 'f'){
            angle = projection_index * coverage / projections;

            p1[0] = distance* sinf(angle);
            p1[1] = distance * cosf(angle);

            p2[0] = -distance * sinf(angle) + u * cosf(angle);
            p2[1] = -distance * cosf(angle) - u * sinf(angle);
        }
        
        float dx = p2[0] - p1[0];
        float dy = p2[1] - p1[1];
        float d_conv = sqrtf(dx * dx + dy * dy);
        float d_ax = (std::abs(dx) > 1e-6f) ? 1.0f / std::abs(dx) : 1e9f;
        float d_ay = (std::abs(dy) > 1e-6f) ? 1.0f / std::abs(dy) : 1e9f;

        float ax_min = (std::abs(dx) > 1e-6f) ? min((bx - p1[0]) / dx, (bx + height - p1[0]) / dx) : -1e9f;
        float ax_max = (std::abs(dx) > 1e-6f) ? max((bx - p1[0]) / dx, (bx + height - p1[0]) / dx) :  1e9f;

        float ay_min = (std::abs(dy) > 1e-6f) ? min((by - p1[1]) / dy, (by + width - p1[1]) / dy) : -1e9f;
        float ay_max = (std::abs(dy) > 1e-6f) ? max((by - p1[1]) / dy, (by + width - p1[1]) / dy) :  1e9f;

        float alpha_min = max(0.0f, max(ax_min, ay_min));
        float alpha_max = min(1.0f, min(ax_max, ay_max));


        float ax_next = 1e9f;
        if (std::abs(dx) > 1e-6f) {
            float rx_entry = p1[0] + alpha_min * dx - bx;
            int next_i = (dx > 0) ? (int)floorf(rx_entry) + 1 : (int)floorf(rx_entry);
            ax_next = (bx + (float)next_i - p1[0]) / dx;
            if (ax_next <= alpha_min) ax_next += d_ax;
        }
        float ay_next = 1e9f;
        if (std::abs(dy) > 1e-6f) {
            float ry_entry = p1[1] + alpha_min * dy - by;
            int next_j = (dy > 0) ? (int)floorf(ry_entry) + 1 : (int)floorf(ry_entry);
            ay_next = (by + (float)next_j - p1[1]) / dy;
            if (ay_next <= alpha_min) ay_next += d_ay;
        }

        float val = sinogram[batch_index * channels * projections * detector_bins + channel_index * projections * detector_bins + projection_index * detector_bins + detector_index];
        float alpha_curr = alpha_min;

        // --- ray march: scatter instead of gather ---
        while (alpha_curr < alpha_max - 1e-6f) {
            float alpha_next = min(alpha_max, min(ax_next, ay_next));
            float mid_alpha = (alpha_curr + alpha_next) * 0.5f;

            int img_i = (int)floorf(p1[0] + mid_alpha * dx - bx);
            int img_j = (int)floorf(p1[1] + mid_alpha * dy - by);

            if (img_i >= 0 && img_i < height && img_j >= 0 && img_j < width) {
                float length = (alpha_next - alpha_curr) * d_conv;
                atomicAdd(&output[batch_index * channels * height * width + channel_index * height * width + img_i * width + img_j], length * val);
            }

            if (ax_next < ay_next) ax_next += d_ax;
            else                   ay_next += d_ay;
            alpha_curr = alpha_next;
        }

    }

torch::Tensor projector_forward(torch::Tensor image, 
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance,
                                char type)
{

    int batches = image.size(0);
    int channels = image.size(1);
    int height = image.size(2);
    int width = image.size(3);

    int threads = 256;
    int blocks_per_projection = (detector_bins + threads - 1) / threads;
    int blocks_per_image = projections * blocks_per_projection;
    int blocks = batches * channels * blocks_per_image;

    auto output = torch::zeros({batches, channels, projections, detector_bins}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));
    
    siddon_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        output.data_ptr<float>(),
        batches,
        channels,
        height,
        width,
        projections,
        detector_bins,
        detector_spacing,
        coverage,
        distance,
        type
    );

    return output;
}


torch::Tensor projector_backward(torch::Tensor sinogram,
                                const std::vector<int64_t>& image_shape, 
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance,
                                char type)
{
    int batches = sinogram.size(0);
    int channels = sinogram.size(1);
    int height = image_shape[2];
    int width = image_shape[3];

    int threads = 256;
    int blocks_per_projection = (detector_bins + threads - 1) / threads;
    int blocks_per_image = projections * blocks_per_projection;
    int blocks = batches * channels * blocks_per_image;

    auto output = torch::zeros({batches, channels, height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(sinogram.device()));
    
    backprojection_kernel<<<blocks, threads>>>(
        sinogram.data_ptr<float>(),
        output.data_ptr<float>(),
        batches,
        channels,
        height,
        width,
        projections,
        detector_bins,
        detector_spacing,
        coverage,
        distance,
        type
    );

    return output;
}