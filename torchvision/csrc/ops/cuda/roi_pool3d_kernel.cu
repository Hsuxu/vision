#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <float.h>
#include <torch/library.h>
#include <THC/THCAtomics.cuh>

#include "cuda_helpers.h"

namespace vision {
namespace ops {

namespace {

template <typename T>
__global__ void roi_pool3d_forward_kernel_impl(
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
    const T* rois,
    T* output,
    int* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pd = (index / pooled_width / pooled_height) % pooled_depth; 
    int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
    int n = index / pooled_width / pooled_height /pooled_depth / channels;

    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];
    int roi_start_d = round(offset_rois[1] * spatial_scale);
    int roi_start_w = round(offset_rois[2] * spatial_scale);
    int roi_start_h = round(offset_rois[3] * spatial_scale);
    int roi_end_d = round(offset_rois[4] * spatial_scale);
    int roi_end_w = round(offset_rois[5] * spatial_scale);
    int roi_end_h = round(offset_rois[6] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_depth = max(roi_end_d - roi_start_d + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(pd) * bin_size_d));
    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int dend = static_cast<int>(ceil(static_cast<T>(pd + 1) * bin_size_d));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    dstart = min(max(dstart + roi_start_d, 0), depth);
    dend = min(max(dend + roi_start_d, 0), depth);
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart)|| (dend <= dstart);

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_input = input + (roi_batch_ind * channels + c) * height * width * depth;
    for (int d = dstart; h < dend; ++d) {
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          int input_index = d * width *height + h * width + w;
          if (offset_input[input_index] > maxval) {
            maxval = offset_input[input_index];
            maxidx = input_index;
          }
        }
      }
    }
    output[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename T>
__global__ void roi_pool3d_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const int* argmax_data,
    int num_rois,
    const T spatial_scale,
    int channels,
    int depth,
    int height,
    int width,
    int pooled_depth,
    int pooled_height,
    int pooled_width,
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
    int c = (index / pooled_width / pooled_height /pooled_depth) % channels;
    int n = index / pooled_width / pooled_height /pooled_depth / channels;

    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];
    T* grad_input_offset = grad_input + ((roi_batch_ind * channels + c) * height * width * depth);

    int output_offset = n * n_stride + c * c_stride;
    const int* argmax_data_offset = argmax_data + (n * channels + c) * pooled_height * pooled_width * pooled_depth;
    int argmax = argmax_data_offset[pd * pooled_height*pooled_width + ph * pooled_width + pw];

    if (argmax != -1) {
      atomicAdd(
          grad_input_offset + argmax,
          static_cast<T>(
              grad_output[output_offset + ph * h_stride + pw * w_stride + pd * d_stride]));
    }
  }
}

std::tuple<at::Tensor, at::Tensor> roi_pool3d_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_depth,
    int64_t pooled_height,
    int64_t pooled_width) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  TORCH_CHECK(rois.size(1) == 7, "Tensor rois should have shape as Tensor[K, 7]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool3d_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto depth = input.size(2);
  auto height = input.size(3);
  auto width = input.size(4);

  at::Tensor output = at::zeros({num_rois, channels, pooled_depth, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros({num_rois, channels, pooled_depth, pooled_height, pooled_width}, input.options().dtype(at::kInt));

  auto output_size = num_rois * pooled_depth * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_pool3d_forward_kernel", [&] {
        roi_pool_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
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
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>());
      });
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

at::Tensor roi_pool3d_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    double spatial_scale,
    int64_t pooled_depth,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t depth,
    int64_t height,
    int64_t width) {
  // Check if input tensors are CUDA tensors
  TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  TORCH_CHECK(argmax.is_cuda(), "argmax must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      argmax_t{argmax, "argmax", 3};

  at::CheckedFrom c = "roi_pool3d_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::cuda::CUDAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);

  at::Tensor grad_input = at::zeros({batch_size, channels, depth, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)), static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int d_stride = grad.stride(2)
  int h_stride = grad.stride(3);
  int w_stride = grad.stride(4);

  auto argmax_ = argmax.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_pool3d_backward_kernel", [&] {
        roi_pool3d_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            argmax_.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            depth,
            height,
            width,
            pooled_depth,
            pooled_height,
            pooled_width,
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
      TORCH_SELECTIVE_NAME("torchvision::roi_pool3d"),
      TORCH_FN(roi_pool3d_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_pool3d_backward"),
      TORCH_FN(roi_pool3d_backward_kernel));
}

} // namespace ops
} // namespace vision
