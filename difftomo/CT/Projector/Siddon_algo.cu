#include<cuda_runtime.h>
#include<torch/extension.h>


#define MAX_PLANES 1024 // Maximum supported grid plane size (nx or ny <= 1024)


__device__ void sort_array(float* arr, int size){
    for (int i = 0; i < size - 1; i++){
        for (int j = 0; j < size - i - 1; j++){
            if (arr[j] > arr[j + 1]){
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }

    return;

}

__device__ void merge(float* arr1, float* arr2, float* merged_array, int size1, int size2){
    int i = 0, j = 0, k = 0;

    while (i < size1 && j < size2){
        if (arr1[i] < arr2[j]){
            merged_array[k++] = arr1[i++];
        }
        else{
            merged_array[k++] = arr2[j++];
        }
    }

    while (i < size1){
        merged_array[k++] = arr1[i++];
    }

    while (j < size2){
        merged_array[k++] = arr2[j++];
    }

    return;
}


__device__ int remove_duplicates(float* arr, int size)
{
    if (size <= 1)
        return size;

    int j = 0;

    for (int i = 1; i < size; i++) {
        if (arr[i] != arr[j]) {
            j++;
            arr[j] = arr[i];
        }
    }

    return j + 1;
}

__global__  
    void siddon_kernel(float* image, float* output, int height, int width, int projections, int detector_bins, float detector_spacing, float coverage, float distance) {

        int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        int blocksPerProjection = ceil((float)detector_bins / (float)threadsPerBlock);

        int projection_index = blockIdx.x / blocksPerProjection;
        int detector_index = blockIdx.x % blocksPerProjection * threadsPerBlock + threadIdx.x;

        if (projection_index >= projections || detector_index >= detector_bins) 
            return;

        float bx = -(height / 2.0f); 
        float by = -(width / 2.0f);
        float u = detector_spacing * (detector_index - detector_bins / 2.0f);

        // thread starts executing here so calculate ray for this detector index and projection index

        float angle = projection_index * coverage / projections;
        float p1[2] = {distance* cosf(angle) - u * sinf(angle),
                       distance* sinf(angle) + u * cosf(angle)};
        float p2[2] = {-distance * cosf(angle) - u * sinf(angle),
                       -distance * sinf(angle) + u * cosf(angle)};
        
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

        int i_min, i_max, j_min, j_max;

        if (std::abs(dx) > 1e-6f) {
            if (dx > 0) {
                i_min = max(0, (int)ceilf(p1[0] + alpha_min * dx - bx));
                i_max = min(height, (int)floorf(p1[0] + alpha_max * dx - bx));
            } else {
                i_min = max(0, (int)ceilf(p1[0] + alpha_max * dx - bx));
                i_max = min(height, (int)floorf(p1[0] + alpha_min * dx - bx));
            }
        } else {
            i_min = 1; i_max = 0;
        }

        if (std::abs(dy) > 1e-6f) {
            if (dy > 0) {
                j_min = max(0, (int)ceilf(p1[1] + alpha_min * dy - by));
                j_max = min(width, (int)floorf(p1[1] + alpha_max * dy - by));
            } else {
                j_min = max(0, (int)ceilf(p1[1] + alpha_max * dy - by));
                j_max = min(width, (int)floorf(p1[1] + alpha_min * dy - by));
            }
        } else {
            j_min = 1; j_max = 0;
        }
        
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
                sum += length * image[img_i * width + img_j];
            }

            if (ax_next < ay_next) {
                ax_next += d_ax;
            } else {
                ay_next += d_ay;
            }
            alpha_curr = alpha_next;
        }

        output[projection_index * detector_bins + detector_index] = sum;
        
    }

__global__ void backprojection_kernel(float* sinogram, float* output, int height, int width, int projections, int detector_bins, float detector_spacing, float coverage, float distance) {
        int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
        int blocksPerProjection = ceil((float)detector_bins / (float)threadsPerBlock);

        int projection_index = blockIdx.x / blocksPerProjection;
        int detector_index = blockIdx.x % blocksPerProjection * threadsPerBlock + threadIdx.x;

        float bx = -(height / 2.0f); 
        float by = -(width / 2.0f);
        float u = detector_spacing * (detector_index - detector_bins / 2.0f);

        // thread starts executing here so calculate ray for this detector index and projection index

        float angle = projection_index * coverage / projections;
        float p1[2] = {distance* cosf(angle) - u * sinf(angle),
                       distance* sinf(angle) + u * cosf(angle)};
        float p2[2] = {-distance * cosf(angle) - u * sinf(angle),
                       -distance * sinf(angle) + u * cosf(angle)};
        
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


        
        // make this a parallelised later
        for(int i=0; i< height; i++){
            float ax0 = (bx + i     - p1[0]) / dx;
            float ax1 = (bx + i + 1 - p1[0]) / dx;
            float ax_initial = min(ax0, ax1);
            float ax_next    = max(ax0, ax1);

            for(int j=0; j<width; j++){
                float ay0 = (by + j     - p1[1]) / dy;
                float ay1 = (by + j + 1 - p1[1]) / dy;

                float ay_initial = min(ay0, ay1);
                float ay_next = ay_initial + d_ay;

                float alpha_curr = max(ax_initial, ay_initial);
                float alpha_next = min(ax_next, ay_next);

                alpha_curr = max(alpha_curr, alpha_min);
                alpha_next = min(alpha_next, alpha_max);
                if (alpha_next > alpha_curr){
                    float length = (alpha_next - alpha_curr) * d_conv;
                    float val = sinogram[projection_index * detector_bins + detector_index];
                    atomicAdd(&output[i * width + j], length * val);
                }

            }

        }

    }

torch::Tensor projector_forward(torch::Tensor image, 
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance)
{

    int blocks = 10000;
    int threads = 256;
    int height = image.size(0);
    int width = image.size(1);
    auto output = torch::zeros({projections, detector_bins}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));

    siddon_kernel<<<blocks, threads>>>(
        image.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width,
        projections,
        detector_bins,
        detector_spacing,
        coverage,
        distance
    );

    return output;
}


torch::Tensor projector_backward(torch::Tensor sinogram,
                                const std::vector<int64_t>& image_shape, 
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance)
{

    int blocks = 10000;
    int threads = 256;
    int height = image_shape[0];
    int width = image_shape[1];
    auto output = torch::zeros({height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(sinogram.device()));

    backprojection_kernel<<<blocks, threads>>>(
        sinogram.data_ptr<float>(),
        output.data_ptr<float>(),
        height,
        width,
        projections,
        detector_bins,
        detector_spacing,
        coverage,
        distance
    );

    return output;
}