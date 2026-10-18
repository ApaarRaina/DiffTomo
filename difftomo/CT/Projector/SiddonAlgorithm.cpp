#include<torch/extension.h>
#include<thread>
#include <vector>
#include <algorithm>

using std::min;
using std::max;


std::mutex image_mutex;

void update_pixel(float* image, int pixel_index, float contribution) {
    std::lock_guard<std::mutex> lock(image_mutex);
    image[pixel_index] += contribution;
}

void forward_siddon(float* image, float* output, int height, int width, int projections, int projection_index, int detector_bins, int num_threads, int thread_index, float detector_spacing, float coverage, float distance, char type){

    for (int offset = 0; offset < detector_bins; offset += num_threads){

        int detector_index = thread_index + offset;

        if (projection_index >= projections || detector_index >= detector_bins){
            return ;
        }

        float bx = -(height / 2.0f); 
        float by = -(width / 2.0f);
        float u = detector_spacing * (detector_index - detector_bins / 2.0f);

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

    return ;
}

void backward_siddon(float* sinogram, float* output, int height, int width, int projections, int projection_index, int detector_bins, int num_threads, int thread_index, float detector_spacing, float coverage, float distance, char type){

    for (int offset = 0; offset < detector_bins; offset += num_threads){
        int detector_index = thread_index + offset;
        
        if (projection_index >= projections || detector_index >= detector_bins) 
            return;

        float bx = -(height / 2.0f); 
        float by = -(width / 2.0f);
        float u = detector_spacing * (detector_index - detector_bins / 2.0f);

        
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

        float val = sinogram[projection_index * detector_bins + detector_index];
        float alpha_curr = alpha_min;

        // --- ray march: scatter instead of gather ---
        while (alpha_curr < alpha_max - 1e-6f) {
            float alpha_next = min(alpha_max, min(ax_next, ay_next));
            float mid_alpha = (alpha_curr + alpha_next) * 0.5f;

            int img_i = (int)floorf(p1[0] + mid_alpha * dx - bx);
            int img_j = (int)floorf(p1[1] + mid_alpha * dy - by);

            if (img_i >= 0 && img_i < height && img_j >= 0 && img_j < width) {
                float length = (alpha_next - alpha_curr) * d_conv;
                update_pixel(output, img_i * width + img_j, length * val);
            }

            if (ax_next < ay_next) ax_next += d_ax;
            else                   ay_next += d_ay;
            alpha_curr = alpha_next;
        }
    }   

    return ;
}




torch::Tensor projector_forward_cpu(torch::Tensor image,
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance,
                                char type,
                                int num_threads)
{
    int height = image.size(0);
    int width = image.size(1);
    auto output = torch::zeros({projections, detector_bins}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));

    for (int projection_index = 0; projection_index < projections; projection_index++){
        std::vector<std::thread> threads;

        for (int thread_index = 0; thread_index < num_threads; thread_index++){
            threads.emplace_back(forward_siddon, image.data_ptr<float>(), output.data_ptr<float>(), height, width, projections, projection_index, detector_bins, num_threads, thread_index, detector_spacing, coverage, distance, type);
        }

        for (auto& t : threads) {
            t.join();
        }   
    }
    
    
    return output;
}

torch::Tensor projector_backward_cpu(torch::Tensor sinogram,
                                const std::vector<int64_t>& image_shape, 
                                int projections, 
                                int detector_bins, 
                                float detector_spacing, 
                                float coverage, 
                                float distance,
                                char type,
                                int num_threads)
{
    int height = image_shape[0];
    int width = image_shape[1];
    auto output = torch::zeros({height, width}, 
                               torch::TensorOptions().dtype(torch::kFloat32).device(sinogram.device()));

    for (int projection_index = 0; projection_index < projections; projection_index++){
        std::vector<std::thread> threads;

        for (int thread_index = 0; thread_index < num_threads; thread_index++){
            threads.emplace_back(backward_siddon, sinogram.data_ptr<float>(), output.data_ptr<float>(), height, width, projections, projection_index, detector_bins, num_threads, thread_index, detector_spacing, coverage, distance, type);
        }

        for (auto& t : threads) {
            t.join();
        }   
    }
    
    
    return output;
}