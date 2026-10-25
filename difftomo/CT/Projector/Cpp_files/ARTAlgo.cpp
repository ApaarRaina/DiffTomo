#include "header_files/Algorithms.h"
#include <algorithm>
#include <thread>
#include <torch/extension.h>
#include <vector>

using std::max;
using std::min;

float aTa_norm(int projections, int detector_bins, float detector_spacing,
               int height, int width, int ray_index, float coverage,
               float distance, char type) {

  int projection_index = ray_index / detector_bins;
  int detector_index = ray_index % detector_bins;

  float bx = -(height / 2.0f);
  float by = -(width / 2.0f);
  float u = detector_spacing * (detector_index - detector_bins / 2.0f);

  // thread starts executing here so calculate ray for this detector index and
  // projection index
  float p1[2];
  float p2[2];
  float angle;

  if (type == 'p') {
    angle = projection_index * coverage / projections;

    p1[0] = distance * cosf(angle) - u * sinf(angle);
    p1[1] = distance * sinf(angle) + u * cosf(angle);

    p2[0] = -distance * cosf(angle) - u * sinf(angle);
    p2[1] = -distance * sinf(angle) + u * cosf(angle);
  }

  else if (type == 'f') {
    angle = projection_index * coverage / projections;

    p1[0] = distance * sinf(angle);
    p1[1] = distance * cosf(angle);

    p2[0] = -distance * sinf(angle) + u * cosf(angle);
    p2[1] = -distance * cosf(angle) - u * sinf(angle);
  }

  float dx = p2[0] - p1[0];
  float dy = p2[1] - p1[1];
  float d_conv = sqrtf(dx * dx + dy * dy);

  // Parametric boundaries for x and y planes
  float ax_min = (std::abs(dx) > 1e-6f)
                     ? min((bx - p1[0]) / dx, (bx + height - p1[0]) / dx)
                     : -1e9f;
  float ax_max = (std::abs(dx) > 1e-6f)
                     ? max((bx - p1[0]) / dx, (bx + height - p1[0]) / dx)
                     : 1e9f;

  float ay_min = (std::abs(dy) > 1e-6f)
                     ? min((by - p1[1]) / dy, (by + width - p1[1]) / dy)
                     : -1e9f;
  float ay_max = (std::abs(dy) > 1e-6f)
                     ? max((by - p1[1]) / dy, (by + width - p1[1]) / dy)
                     : 1e9f;

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
    if (ax_next <= alpha_min)
      ax_next += d_ax;
  }

  float ay_next = 1e9f;
  if (std::abs(dy) > 1e-6f) {
    float ry_entry = p1[1] + alpha_min * dy - by;
    int next_j = (dy > 0) ? (int)floorf(ry_entry) + 1 : (int)floorf(ry_entry);
    ay_next = (by + (float)next_j - p1[1]) / dy;
    if (ay_next <= alpha_min)
      ay_next += d_ay;
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
      sum += length * length;
    }

    if (ax_next < ay_next) {
      ax_next += d_ax;
    } else {
      ay_next += d_ay;
    }
    alpha_curr = alpha_next;
  }

  return sum;
}

float aTx_k(float *image, int projections, int detector_bins,
            float detector_spacing, int batches, int channels, int height,
            int width, int batch_index, int channel_index, int ray_index,
            float coverage, float distance, char type) {

  int projection_index = ray_index / detector_bins;
  int detector_index = ray_index % detector_bins;
  float bx = -(height / 2.0f);
  float by = -(width / 2.0f);
  float u = detector_spacing * (detector_index - detector_bins / 2.0f);

  // thread starts executing here so calculate ray for this detector index and
  // projection index
  float p1[2];
  float p2[2];
  float angle;

  if (type == 'p') {
    angle = projection_index * coverage / projections;

    p1[0] = distance * cosf(angle) - u * sinf(angle);
    p1[1] = distance * sinf(angle) + u * cosf(angle);

    p2[0] = -distance * cosf(angle) - u * sinf(angle);
    p2[1] = -distance * sinf(angle) + u * cosf(angle);
  }

  else if (type == 'f') {
    angle = projection_index * coverage / projections;

    p1[0] = distance * sinf(angle);
    p1[1] = distance * cosf(angle);

    p2[0] = -distance * sinf(angle) + u * cosf(angle);
    p2[1] = -distance * cosf(angle) - u * sinf(angle);
  }

  float dx = p2[0] - p1[0];
  float dy = p2[1] - p1[1];
  float d_conv = sqrtf(dx * dx + dy * dy);

  // Parametric boundaries for x and y planes
  float ax_min = (std::abs(dx) > 1e-6f)
                     ? min((bx - p1[0]) / dx, (bx + height - p1[0]) / dx)
                     : -1e9f;
  float ax_max = (std::abs(dx) > 1e-6f)
                     ? max((bx - p1[0]) / dx, (bx + height - p1[0]) / dx)
                     : 1e9f;

  float ay_min = (std::abs(dy) > 1e-6f)
                     ? min((by - p1[1]) / dy, (by + width - p1[1]) / dy)
                     : -1e9f;
  float ay_max = (std::abs(dy) > 1e-6f)
                     ? max((by - p1[1]) / dy, (by + width - p1[1]) / dy)
                     : 1e9f;

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
    if (ax_next <= alpha_min)
      ax_next += d_ax;
  }

  float ay_next = 1e9f;
  if (std::abs(dy) > 1e-6f) {
    float ry_entry = p1[1] + alpha_min * dy - by;
    int next_j = (dy > 0) ? (int)floorf(ry_entry) + 1 : (int)floorf(ry_entry);
    ay_next = (by + (float)next_j - p1[1]) / dy;
    if (ay_next <= alpha_min)
      ay_next += d_ay;
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
      sum += length *
             image[batch_index * channels * height * width +
                   channel_index * height * width + img_i * width + img_j];
    }

    if (ax_next < ay_next) {
      ax_next += d_ax;
    } else {
      ay_next += d_ay;
    }
    alpha_curr = alpha_next;
  }

  return sum;
}

void y_ai(float val, float *output, int projections, int detector_bins,
          float detector_spacing, int channels, int height, int width,
          int batch_index, int channel_index, int ray_index, float coverage,
          float distance, char type) {
  int projection_index = ray_index / detector_bins;
  int detector_index = ray_index % detector_bins;

  // ray origin
  float bx = -(height / 2.0f);
  float by = -(width / 2.0f);
  float u = detector_spacing * (detector_index - detector_bins / 2.0f);

  // thread starts executing here so calculate ray for this detector index and
  // projection index
  float p1[2];
  float p2[2];
  float angle;

  if (type == 'p') {
    angle = projection_index * coverage / projections;

    p1[0] = distance * cosf(angle) - u * sinf(angle);
    p1[1] = distance * sinf(angle) + u * cosf(angle);

    p2[0] = -distance * cosf(angle) - u * sinf(angle);
    p2[1] = -distance * sinf(angle) + u * cosf(angle);
  }

  else if (type == 'f') {
    angle = projection_index * coverage / projections;

    p1[0] = distance * sinf(angle);
    p1[1] = distance * cosf(angle);

    p2[0] = -distance * sinf(angle) + u * cosf(angle);
    p2[1] = -distance * cosf(angle) - u * sinf(angle);
  }

  float dx = p2[0] - p1[0];
  float dy = p2[1] - p1[1];
  float d_conv = sqrtf(dx * dx + dy * dy);
  float d_ax = (std::abs(dx) > 1e-6f) ? 1.0f / std::abs(dx) : 1e9f;
  float d_ay = (std::abs(dy) > 1e-6f) ? 1.0f / std::abs(dy) : 1e9f;

  float ax_min = (std::abs(dx) > 1e-6f)
                     ? min((bx - p1[0]) / dx, (bx + height - p1[0]) / dx)
                     : -1e9f;
  float ax_max = (std::abs(dx) > 1e-6f)
                     ? max((bx - p1[0]) / dx, (bx + height - p1[0]) / dx)
                     : 1e9f;

  float ay_min = (std::abs(dy) > 1e-6f)
                     ? min((by - p1[1]) / dy, (by + width - p1[1]) / dy)
                     : -1e9f;
  float ay_max = (std::abs(dy) > 1e-6f)
                     ? max((by - p1[1]) / dy, (by + width - p1[1]) / dy)
                     : 1e9f;

  float alpha_min = max(0.0f, max(ax_min, ay_min));
  float alpha_max = min(1.0f, min(ax_max, ay_max));

  float ax_next = 1e9f;
  if (std::abs(dx) > 1e-6f) {
    float rx_entry = p1[0] + alpha_min * dx - bx;
    int next_i = (dx > 0) ? (int)floorf(rx_entry) + 1 : (int)floorf(rx_entry);
    ax_next = (bx + (float)next_i - p1[0]) / dx;
    if (ax_next <= alpha_min)
      ax_next += d_ax;
  }
  float ay_next = 1e9f;
  if (std::abs(dy) > 1e-6f) {
    float ry_entry = p1[1] + alpha_min * dy - by;
    int next_j = (dy > 0) ? (int)floorf(ry_entry) + 1 : (int)floorf(ry_entry);
    ay_next = (by + (float)next_j - p1[1]) / dy;
    if (ay_next <= alpha_min)
      ay_next += d_ay;
  }

  float sum = 0.0f;
  float alpha_curr = alpha_min;

  // --- ray march: scatter instead of gather ---
  while (alpha_curr < alpha_max - 1e-6f) {
    float alpha_next = min(alpha_max, min(ax_next, ay_next));
    float mid_alpha = (alpha_curr + alpha_next) * 0.5f;

    int img_i = (int)floorf(p1[0] + mid_alpha * dx - bx);
    int img_j = (int)floorf(p1[1] + mid_alpha * dy - by);

    if (img_i >= 0 && img_i < height && img_j >= 0 && img_j < width) {
      float length = (alpha_next - alpha_curr) * d_conv;
      output[batch_index * channels * height * width +
             channel_index * height * width + img_i * width + img_j] +=
          length * val;
    }

    if (ax_next < ay_next)
      ax_next += d_ax;
    else
      ay_next += d_ay;
    alpha_curr = alpha_next;
  }

  return;
}

torch::Tensor ARTReconstruction(torch::Tensor sinogram,
                                const std::vector<int64_t> &image_shape,
                                int projections, int detector_bins,
                                int iterations, float relaxation_parameter,
                                float detector_spacing, float coverage,
                                float distance, char type)

{
  int batches = sinogram.size(0);
  int channels = sinogram.size(1);
  int height = image_shape[0];
  int width = image_shape[1];
  float eps = 1e-6f;
  float norm, lam_n, residual;
  auto output_k = torch::zeros({batches, channels, height, width},
                               torch::TensorOptions().dtype(torch::kFloat32));

  float *sinogram_ptr = sinogram.data_ptr<float>();
  int n_rays = projections * detector_bins;
  std::vector<float> norms(n_rays);
  for (int r = 0; r < n_rays; ++r)
    norms[r] = aTa_norm(projections, detector_bins, detector_spacing, height,
                        width, r, coverage, distance, type);

  for (int iter = 0; iter < iterations; ++iter) {
    for (int i = 0; i < batches * channels * projections * detector_bins; ++i) {
      int batch_index = i / (channels * projections * detector_bins);
      int channel_index = (i / (projections * detector_bins)) % channels;
      int ray_index = i % (projections * detector_bins);
      norm = norms[ray_index];
      if (norm < eps) {
        continue;
      }
      lam_n = relaxation_parameter / norm;
      float aTx =
          aTx_k(output_k.data_ptr<float>(), projections, detector_bins,
                detector_spacing, batches, channels, height, width, batch_index,
                channel_index, ray_index, coverage, distance, type);
      float scale = lam_n * (sinogram_ptr[i] - aTx);
      y_ai(scale, output_k.data_ptr<float>(), projections, detector_bins,
           detector_spacing, channels, height, width, batch_index,
           channel_index, ray_index, coverage, distance, type);
    }
  }

  return output_k;
}