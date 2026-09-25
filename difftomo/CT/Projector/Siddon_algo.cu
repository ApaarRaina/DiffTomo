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

__device__ void remove_duplicates(float* arr, int size) {
    if (size <= 1) return;

    int j = 0;

    for (int i = 1; i < size; i++) {
        if (arr[i] - arr[j] > 0) {
            j++;
            arr[j] = arr[i];
        }
    }

    return;
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

        // thread starts executing here so calculate ray for this detector index and projection index

        float angle = projection_index * coverage / projections;
        float p1[2] = {distance* cosf(angle) - detector_spacing * (detector_index - detector_bins / 2) * sinf(angle),
                       distance* sinf(angle) + detector_spacing * (detector_index - detector_bins / 2) * cosf(angle)};
        float p2[2] = {-distance * cosf(angle) + detector_spacing * (detector_index - detector_bins / 2) * sinf(angle),
                       -distance * sinf(angle) - detector_spacing * (detector_index - detector_bins / 2) * cosf(angle)};
        
        float d_conv = sqrtf(((p2[0] - p1[0])*(p2[0] - p1[0])) + ((p2[1] - p1[1])*(p2[1] - p1[1])));
        
        float alpha_array_x[MAX_PLANES], alpha_array_y[MAX_PLANES];
        float alpha_array_xy[MAX_PLANES * 2];
        
        for (int i = 0; i < height; i++){
            if (p1[0] == p2[0]){
                alpha_array_x[i] = 0
                continue;
            }
            alpha_array_x[i] = (bx + i - p1[0]) / (p2[0] - p1[0]);
            
        }
        sort_array(alpha_array_x, height)

        for (int i = 0; i < width; i++){
            if (p1[1] == p2[1]){
                alpha_array_y[i] = 0
                continue;
            }
            alpha_array_y[i] = (by + i - p1[1]) / (p2[1] - p1[1]);
        }
        sort_array(alpha_array_y, width)

        float alpha_min = max(0, max(alpha_array_x[0], alpha_array_y[0]));
        float alpha_max = min(1, min(alpha_array_x[height - 1], alpha_array_y[width - 1]));

        alpha_array_xy  = merge(alpha_array_x, alpha_array_y, alpha_array_xy, height, width);
        float l;
        int i_m, j_m;
        remove_duplicates(alpha_array_xy, height + width);


        if (alpha_min < alpha_max){
            float sum = 0;
            int mid;
            for (int m = 0; m < height + width - 1; m++){
                mid = (alpha_array_xy[m] + alpha_array_xy[m + 1]) / 2;
                i_m = floor(p1[0] + mid * (p2[0] - p1[0]) - bx);
                j_m = floor(p1[1] + mid * (p2[1] - p1[1]) - by);
                l = (alpha_array_xy[m + 1] - alpha_array_xy[m]) * d_conv;
                sum += l * image[i_m][j_m];
            }
            output[projection_index][detector_index] = sum;
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

    forward_kernel<<<blocks, threads>>>(
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