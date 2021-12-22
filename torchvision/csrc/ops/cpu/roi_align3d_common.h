#pragma once

#include <ATen/ATen.h>

namespace vision {
namespace ops {
namespace detail {

template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  int pos5;
  int pos6;
  int pos7;
  int pos8;
  T w1;
  T w2;
  T w3;
  T w4;
  T w5;
  T w6;
  T w7;
  T w8;
};

// This helper computes the interpolation weights (w1, w2...) for every sampling
// point of a given box. There are   * pool_width * roi_bin_grid_h *
// roi_bin_grid_w such sampling points.
//
// The weights (w1, w2...) are computed as the areas in this figure:
// https://en.wikipedia.org/wiki/Bilinear_interpolation#/media/File:Bilinear_interpolation_visualisation.svg
// and pos1, pos2 etc correspond to the indices of their respective pixels.
//
// Note: the weights and indices are shared across all channels, which is why
// they are pre-calculated prior to the main loop in the RoIAlign kernel.
// implementation taken from Caffe2
template <typename T>
void pre_calc_for_trilinear_interpolate(
    int depth,
    int height,
    int width,
    int pooled_depth,
    int pooled_height,
    int pooled_width,
    T roi_start_d,
    T roi_start_h,
    T roi_start_w,
    T bin_size_d,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_d,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int pd = 0; pd < pooled_depth; pd++) 
  {
    for (int ph = 0; ph < pooled_height; ph++) 
    {
      for (int pw = 0; pw < pooled_width; pw++) 
      {
        for (int iz = 0; iz < roi_bin_grid_d; iz++)
        {
          const T zz = roi_start_d + pd * bin_size_d + static_cast<T>(iz + .5f) * bin_size_d /static_cast<T>(roi_bin_grid_d); // e.g., 0.5, 1.5

          for (int iy = 0; iy < roi_bin_grid_h; iy++) 
          {
            const T yy = roi_start_h + ph * bin_size_h + static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5

            for (int ix = 0; ix < roi_bin_grid_w; ix++) 
            {
              const T xx = roi_start_w + pw * bin_size_w +
                  static_cast<T>(ix + .5f) * bin_size_w /
                      static_cast<T>(roi_bin_grid_w);

              T x = xx;
              T y = yy;
              T z = zz;
              // deal with: inverse elements are out of feature map boundary
              if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                PreCalc<T> pc;
                pc.pos1 = 0;
                pc.pos2 = 0;
                pc.pos3 = 0;
                pc.pos4 = 0;
                pc.pos5 = 0;
                pc.pos6 = 0;
                pc.pos7 = 0;
                pc.pos8 = 0;
                pc.w1 = 0;
                pc.w2 = 0;
                pc.w3 = 0;
                pc.w4 = 0;
                pc.w5 = 0;
                pc.w6 = 0;
                pc.w7 = 0;
                pc.w8 = 0;
                pre_calc[pre_calc_index] = pc;
                pre_calc_index += 1;
                continue;
              }

              if (z <= 0) {
                z = 0;
              }
              if (y <= 0) {
                y = 0;
              }
              if (x <= 0) {
                x = 0;
              }

              int z_low = (int)z;
              int y_low = (int)y;
              int x_low = (int)x;
              int z_high;
              int y_high;
              int x_high;

              if (z_low >= depth - 1) {
                z_high = z_low = depth - 1;
                z = (T)z_low;
              } else {
                z_high = z_low + 1;
              }

              if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (T)y_low;
              } else {
                y_high = y_low + 1;
              }

              if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (T)x_low;
              } else {
                x_high = x_low + 1;
              }

              T lz = z - z_low;
              T ly = y - y_low;
              T lx = x - x_low;

              T hz = 1. - lz, hy = 1. - ly, hx = 1. - lx;
              T w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
              T w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;

              // save weights and indices
              PreCalc<T> pc;
              pc.pos1 = z_low * height * width + y_low * width + x_low;
              pc.pos2 = z_low * height * width + y_low * width + x_high;
              pc.pos3 = z_low * height * width + y_high * width + x_low;
              pc.pos4 = z_low * height * width + y_high * width + x_high;
              pc.pos5 = z_high * height * width + y_low * width + x_low;
              pc.pos6 = z_high * height * width + y_low * width + x_high;
              pc.pos7 = z_high * height * width + y_high * width + x_low;
              pc.pos8 = z_high * height * width + y_high * width + x_high;
              pc.w1 = w1;
              pc.w2 = w2;
              pc.w3 = w3;
              pc.w4 = w4;
              pc.w5 = w5;
              pc.w6 = w6;
              pc.w7 = w7;
              pc.w8 = w8;
              pre_calc[pre_calc_index] = pc;

              pre_calc_index += 1;
            }
          }
        }
      }
    }
  }
}

} // namespace detail
} // namespace ops
} // namespace vision
