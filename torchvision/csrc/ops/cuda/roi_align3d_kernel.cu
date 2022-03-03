#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

template <typename T>
__device__ T trilinear_interpolate(
    const T* input,
    int depth,
    int height,
    int width,
    T z,
    T y,
    T x,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (z <= 0)
    z = 0;
  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

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

  // do triilinear interpolation
  T v1 = input[z_low * height * width + y_low * width + x_low];
  T v2 = input[z_low * height * width +y_low * width + x_high];
  T v3 = input[z_low * height * width +y_high * width + x_low];
  T v4 = input[z_low * height * width +y_high * width + x_high];
  T v5 = input[z_high * height * width + y_low * width + x_low];
  T v6 = input[z_high * height * width + y_low * width + x_high];
  T v7 = input[z_high * height * width + y_high * width + x_low];
  T v8 = input[z_high * height * width + y_high * width + x_high];

  T w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
  T w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 *v6 + w7 * v7 + w8 * v8);

  return val;
}

template <typename T>
__global__ void roi_align3d_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int depth,
    int height,
    int width,
    int pooled_depth,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    const T* rois,
    T* output) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pd, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_height/pooled_width) % pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth/ channels;

    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_d = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_start_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_d = offset_rois[4] * spatial_scale - offset;
    T roi_end_h = offset_rois[5] * spatial_scale - offset;
    T roi_end_w = offset_rois[6] * spatial_scale - offset;

    T roi_depth = roi_end_d - roi_start_d;
    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_depth = max(roi_depth,(T)1.);
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width * depth;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_d =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros.
    const T count = max(roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d, 1); // e.g. = 4

    T output_val = 0.;
    for (int iz = 0; iz < roi_bin_grid_d; iz++)
    {
      const T z = roi_start_d + pd * bin_size_d +
            static_cast<T>(iz + .5f) * bin_size_d /
                static_cast<T>(roi_bin_grid_d);
      for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
      {
        const T y = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) 
        {
          const T x = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T val = trilinear_interpolate(offset_input, depth, height, width, z, y, x, index);
          output_val += val;
        } // ix
      } //iy
    } // iz
    output_val /= count;
    output[index] = output_val;
  }
}

template <typename T>
__device__ void trilinear_interpolate_gradient(
    int depth,
    int height,
    int width,
    T z,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    T& w5,
    T& w6,
    T& w7,
    T& w8,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    int& z_low,
    int& z_high,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (z < -1.0 || z > depth || y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    w5 = w6 = w7 = w8 = 0.;
    x_low = x_high = y_low = y_high = z_low = z_high = -1;
    return;
  }

  if (z <= 0)
    z = 0;
  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  z_low = (int)z;
  y_low = (int)y;
  x_low = (int)x;

  if (z_low >= depth - 1) {
    z_high = z_low = depth - 1;
    z = (T)z_high;
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

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
  w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;
}

template <typename T>
__global__ void roi_align3d_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const T spatial_scale,
    int channels,
    int depth,
    int height,
    int width,
    int pooled_depth,
    int pooled_height,
    int pooled_width,
    int sampling_ratio,
    bool aligned,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int d_stride,
    int h_stride,
    int w_stride) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth;
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height / pooled_depth/ channels;

    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T roi_start_d = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_start_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_d = offset_rois[4] * spatial_scale - offset;
    T roi_end_h = offset_rois[5] * spatial_scale - offset;
    T roi_end_w = offset_rois[6] * spatial_scale - offset;

    T roi_depth = roi_end_d - roi_start_d;
    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if (!aligned) {
      // Force malformed ROIs to be 1x1
      roi_depth = max(roi_depth, (T)1.);
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width * depth);

    // We need to index the gradient using the tensor strides to access the
    // correct values.
    int output_offset = n * n_stride + c * c_stride;
    const T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride + pd * d_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_d =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_depth / pooled_depth);
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w * roi_bin_grid_d; // e.g. = 4
    
    for (int iz = 0; iz < roi_bin_grid_d; iz++) // e.g., iy = 0, 1
    {
      const T z = roi_start_d + pd * bin_size_d +
            static_cast<T>(iz + .5f) * bin_size_d /
                static_cast<T>(roi_bin_grid_d);
      for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
      {
        const T y = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          const T x = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T w1, w2, w3, w4, w5, w6, w7, w8;
          int x_low, x_high, y_low, y_high, z_low, z_high;

          trilinear_interpolate_gradient(
              depth,
              height,
              width,
              z,
              y,
              x,
              w1,
              w2,
              w3,
              w4,
              w5, 
              w6, 
              w7, 
              w8,
              x_low,
              x_high,
              y_low,
              y_high,
              z_low,
              z_high,
              index);

          T g1 = grad_output_this_bin * w1 / count;
          T g2 = grad_output_this_bin * w2 / count;
          T g3 = grad_output_this_bin * w3 / count;
          T g4 = grad_output_this_bin * w4 / count;
          T g5 = grad_output_this_bin * w5 / count;
          T g6 = grad_output_this_bin * w6 / count;
          T g7 = grad_output_this_bin * w7 / count;
          T g8 = grad_output_this_bin * w8 / count;

          if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && z_low >= 0 && z_high >= 0) {
            atomicAdd(
                offset_grad_input + z_low * height * width + y_low * width + x_low, static_cast<T>(g1));
            atomicAdd(
                offset_grad_input + z_low * height * width + y_low * width + x_high, static_cast<T>(g2));
            atomicAdd(
                offset_grad_input + z_low * height * width + y_high * width + x_low, static_cast<T>(g3));
            atomicAdd(
                offset_grad_input + z_low * height * width + y_high * width + x_high, static_cast<T>(g4));
            atomicAdd(
                offset_grad_input + z_high * height * width + y_low * width + x_low, static_cast<T>(g5));
            atomicAdd(
                offset_grad_input + z_high * height * width + y_low * width + x_high, static_cast<T>(g6));
            atomicAdd(
                offset_grad_input + z_high * height * width + y_high * width + x_low, static_cast<T>(g7));
            atomicAdd(
                offset_grad_input + z_high * height * width + y_high * width + x_high, static_cast<T>(g8));
          } // if
        } // ix
      } // iy
    } //iz
  } // CUDA_1D_KERNEL_LOOP
}

at::Tensor roi_align3d_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_depth,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  TORCH_CHECK(rois.size(1) == 7, "rois must have shape as Tensor[K, 7]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align3d_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto depth = input.size(2);
  auto height = input.size(3);
  auto width = input.size(4);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_depth, pooled_height, pooled_width}, input.options());

  auto output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_align3d_forward_kernel", [&] {
        roi_align3d_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            depth,
            height,
            width,
            pooled_depth,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

at::Tensor roi_align3d_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_depth,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t depth,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align3d_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::cuda::CUDAGuard device_guard(grad.device());

  at::Tensor grad_input =
      at::zeros({batch_size, channels, depth, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int d_stride = grad.stride(2);
  int h_stride = grad.stride(3);
  int w_stride = grad.stride(4);

  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_align3d_backward_kernel", [&] {
        roi_align3d_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            depth,
            height,
            width,
            pooled_depth,
            pooled_height,
            pooled_width,
            sampling_ratio,
            aligned,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            d_stride,
            h_stride,
            w_stride);
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align3d"),
      TORCH_FN(roi_align3d_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_align3d_backward"),
      TORCH_FN(roi_align3d_backward_kernel));
}

} // namespace ops
} // namespace vision
