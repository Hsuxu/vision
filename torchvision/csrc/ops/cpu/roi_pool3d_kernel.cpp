#include <float.h>

#include <ATen/ATen.h>
#include <torch/library.h>

namespace vision {
namespace ops {

namespace {

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename T>
void roi_pool3d_forward_kernel_impl(
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
    int num_rois,
    T* output,
    int* argmax_data) {
  for (int n = 0; n < num_rois; ++n) {
    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];
    int roi_start_d = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_start_w = round(offset_rois[3] * spatial_scale);
    int roi_end_d = round(offset_rois[4] * spatial_scale);
    int roi_end_h = round(offset_rois[5] * spatial_scale);
    int roi_end_w = round(offset_rois[6] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_depth = std::max(roi_end_d - roi_start_d + 1, 1);
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    T bin_size_d = static_cast<T>(roi_depth) / static_cast<T>(pooled_depth);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    for (int pd = 0; pd < pooled_depth; ++pd) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        for (int pw = 0; pw < pooled_width; ++pw) {
          int dstart = static_cast<int>(floor(static_cast<T>(pd) * bin_size_d));
          int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
          int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
          int dend =
              static_cast<int>(ceil(static_cast<T>(pd + 1) * bin_size_d));
          int hend =
              static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
          int wend =
              static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

          // Add roi offsets and clip to input boundaries
          dstart = std::min(std::max(dstart + roi_start_d, 0), depth);
          dend = std::min(std::max(dend + roi_start_d, 0), depth);
          hstart = std::min(std::max(hstart + roi_start_h, 0), height);
          hend = std::min(std::max(hend + roi_start_h, 0), height);
          wstart = std::min(std::max(wstart + roi_start_w, 0), width);
          wend = std::min(std::max(wend + roi_start_w, 0), width);
          bool is_empty =
              (hend <= hstart) || (wend <= wstart) || dend <= dstart;

          for (int c = 0; c < channels; ++c) {
            // Define an empty pooling region to be zero
            T maxval = is_empty ? 0 : -FLT_MAX;
            // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
            int maxidx = -1;

            const T* input_offset =
                input + (roi_batch_ind * channels + c) * height * width * depth;

            for (int d = dstart; d < dend; ++d) {
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  int input_index = d * height * width + h * width + w;
                  if (input_offset[input_index] > maxval) {
                    maxval = input_offset[input_index];
                    maxidx = input_index;
                  }
                }
              }
            }
            int index =
                (((n * channels + c) * pooled_depth + pd) * pooled_height +
                 ph) *
                    pooled_width +
                pw;
            output[index] = maxval;
            argmax_data[index] = maxidx;
          } // channels
        } // pooled_width
      } // pooled_height
    } // pooled_depth
  } // num_rois
}

template <typename T>
void roi_pool3d_backward_kernel_impl(
    const T* grad_output,
    const int* argmax_data,
    int num_rois,
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
  for (int n = 0; n < num_rois; ++n) {
    const T* offset_rois = rois + n * 7;
    int roi_batch_ind = offset_rois[0];

    for (int c = 0; c < channels; ++c) {
      T* grad_input_offset = grad_input +
          ((roi_batch_ind * channels + c) * depth * height * width);
      const int* argmax_data_offset = argmax_data +
          (n * channels + c) * pooled_depth * pooled_height * pooled_width;

      for (int pd = 0; pd < pooled_depth; ++pd) {
        for (int ph = 0; ph < pooled_height; ++ph) {
          for (int pw = 0; pw < pooled_width; ++pw) {
            int output_offset = n * n_stride + c * c_stride;
            int argmax = argmax_data_offset
                [pd * pooled_height * pooled_width + ph * pooled_width + pw];

            if (argmax != -1) {
              add(grad_input_offset + argmax,
                  static_cast<T>(grad_output
                                     [output_offset + pd * d_stride +
                                      ph * h_stride + pw * w_stride]));
            }
          } // pooled_width
        } // pooled_height
      } // pooled_depth
    } // channels
  } // num_rois
}

std::tuple<at::Tensor, at::Tensor> roi_pool3d_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_depth,
    int64_t pooled_height,
    int64_t pooled_width) {
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");
  TORCH_CHECK(rois.device().is_cpu(), "rois must be a CPU tensor");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool3d_forward_kernel";
  at::checkAllSameType(c, {input_t, rois_t});

  int num_rois = rois.size(0);
  int channels = input.size(1);
  int depth = input.size(2);
  int height = input.size(3);
  int width = input.size(4);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_depth, pooled_height, pooled_width},
      input.options());
  at::Tensor argmax = at::zeros(
      {num_rois, channels, pooled_depth, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  if (output.numel() == 0) {
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_pool3d_forward_kernel", [&] {
        roi_pool3d_forward_kernel_impl<scalar_t>(
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
            num_rois,
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>());
      });
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
  // Check if input tensors are CPU tensors
  TORCH_CHECK(grad.device().is_cpu(), "grad must be a CPU tensor");
  TORCH_CHECK(rois.device().is_cpu(), "rois must be a CPU tensor");
  TORCH_CHECK(argmax.device().is_cpu(), "argmax must be a CPU tensor");
  TORCH_CHECK(
      rois.size(1) == 7, "Tensor rois should have shape as Tensor[K, 7]");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool3d_backward_kernel";
  at::checkAllSameType(c, {grad_t, rois_t});

  auto num_rois = rois.size(0);

  at::Tensor grad_input =
      at::zeros({batch_size, channels, depth, height, width}, grad.options());

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  // get stride values to ensure indexing into gradients is correct.
  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int d_stride = grad.stride(2);
  int h_stride = grad.stride(3);
  int w_stride = grad.stride(4);

  auto rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_pool3d_backward_kernel", [&] {
        roi_pool3d_backward_kernel_impl<scalar_t>(
            grad.data_ptr<scalar_t>(),
            argmax.data_ptr<int>(),
            num_rois,
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
  return grad_input;
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool3d"),
      TORCH_FN(roi_pool3d_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_pool3d_backward"),
      TORCH_FN(roi_pool3d_backward_kernel));
}

} // namespace ops
} // namespace vision
