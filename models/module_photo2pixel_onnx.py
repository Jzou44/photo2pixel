import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConfigurablePhoto2PixelOnnxModel(nn.Module):
    def __init__(self, num_bins=4, edge_dilate=3):
        super(ConfigurablePhoto2PixelOnnxModel, self).__init__()
        self.num_bins = num_bins
        self.edge_dilate = edge_dilate

        kernel_sobel_h = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_h = torch.from_numpy(kernel_sobel_h).repeat([3, 1, 1, 1])
        self.register_buffer("kernel_sobel_h", kernel_sobel_h)

        kernel_sobel_v = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_v = torch.from_numpy(kernel_sobel_v).repeat([3, 1, 1, 1])
        self.register_buffer("kernel_sobel_v", kernel_sobel_v)

    @staticmethod
    def _scalar(value, dtype, device):
        if torch.is_tensor(value):
            return value.to(dtype=dtype, device=device).reshape(())
        return torch.tensor(value, dtype=dtype, device=device)

    @staticmethod
    def _gather_bins(data, indices):
        return torch.gather(data, dim=0, index=indices.unsqueeze(0)).squeeze(0)

    @staticmethod
    def _arange(end, device):
        if not torch.jit.is_tracing() and torch.is_tensor(end):
            end = int(end.item())
        return torch.arange(end, device=device, dtype=torch.int64)

    @staticmethod
    def _clamp_indices(indices, max_value):
        zero = torch.zeros((), dtype=torch.int64, device=indices.device)
        return torch.minimum(torch.maximum(indices, zero), max_value)

    @staticmethod
    def _nearest_resize_by_factor(rgb, factor):
        shape = torch._shape_as_tensor(rgb)
        height = shape[-2]
        width = shape[-1]
        target_h = height * factor
        target_w = width * factor

        y = torch.div(ConfigurablePhoto2PixelOnnxModel._arange(target_h, rgb.device), factor, rounding_mode="floor")
        x = torch.div(ConfigurablePhoto2PixelOnnxModel._arange(target_w, rgb.device), factor, rounding_mode="floor")
        return rgb.index_select(2, y).index_select(3, x)

    def _window_sums(self, data, kernel_size, pixel_size):
        shape = torch._shape_as_tensor(data)
        height = shape[-2]
        width = shape[-1]
        kernel_pad = torch.div(kernel_size - 1, 2, rounding_mode="floor")

        conv_h = height + 2 * kernel_pad - kernel_size + 1
        conv_w = width + 2 * kernel_pad - kernel_size + 1
        out_h = torch.div(conv_h - 1, pixel_size, rounding_mode="floor") + 1
        out_w = torch.div(conv_w - 1, pixel_size, rounding_mode="floor") + 1
        y = self._arange(out_h, data.device) * pixel_size
        x = self._arange(out_w, data.device) * pixel_size

        y0 = self._clamp_indices(y - kernel_pad, height)
        y1 = self._clamp_indices(y - kernel_pad + kernel_size, height)
        x0 = self._clamp_indices(x - kernel_pad, width)
        x1 = self._clamp_indices(x - kernel_pad + kernel_size, width)

        integral = torch.cumsum(torch.cumsum(data, dim=2), dim=3)
        integral = F.pad(integral, (1, 0, 1, 0))

        bottom_right = integral.index_select(2, y1).index_select(3, x1)
        top_right = integral.index_select(2, y0).index_select(3, x1)
        bottom_left = integral.index_select(2, y1).index_select(3, x0)
        top_left = integral.index_select(2, y0).index_select(3, x0)
        return bottom_right - top_right - bottom_left + top_left

    def _pixel_effect(self, rgb, kernel_size, pixel_size):
        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        intensity_idx = torch.mean(rgb, dim=[0, 1]) / 256. * self.num_bins
        intensity_idx = torch.clamp(intensity_idx.long(), min=0, max=self.num_bins - 1)
        intensity = F.one_hot(intensity_idx, num_classes=self.num_bins)
        intensity = torch.permute(intensity, dims=[2, 0, 1]).unsqueeze(dim=0).to(dtype=rgb.dtype)

        weighted = torch.cat([r * intensity, g * intensity, b * intensity, intensity], dim=1)
        summed = self._window_sums(weighted, kernel_size, pixel_size)[0, :, :, :]

        r_sum, g_sum, b_sum, intensity = torch.chunk(summed, chunks=4, dim=0)
        intensity_max, intensity_argmax = torch.max(intensity, dim=0)
        intensity_max = torch.clamp(intensity_max, min=1)

        r = self._gather_bins(r_sum, intensity_argmax) / intensity_max
        g = self._gather_bins(g_sum, intensity_argmax) / intensity_max
        b = self._gather_bins(b_sum, intensity_argmax) / intensity_max

        result_rgb = torch.stack([r, g, b], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)
        return self._nearest_resize_by_factor(result_rgb, pixel_size)

    def _edge_detect(self, rgb, edge_thresh):
        rgb = F.pad(rgb, (1, 1, 1, 1), mode="reflect")
        edge_h = F.conv2d(rgb, self.kernel_sobel_h, groups=3)
        edge_w = F.conv2d(rgb, self.kernel_sobel_v, groups=3)
        edge = torch.stack([torch.abs(edge_h), torch.abs(edge_w)], dim=1)
        edge = torch.max(edge, dim=1)[0]
        edge = torch.mean(edge, dim=1, keepdim=True)
        edge = torch.gt(edge, edge_thresh).to(dtype=rgb.dtype)
        return F.max_pool2d(edge, kernel_size=self.edge_dilate, stride=1, padding=self.edge_dilate // 2)

    def forward(self, rgb, param_kernel_size, param_pixel_size, param_edge_thresh):
        kernel_size = self._scalar(param_kernel_size, torch.int64, rgb.device)
        pixel_size = self._scalar(param_pixel_size, torch.int64, rgb.device)
        edge_thresh = self._scalar(param_edge_thresh, rgb.dtype, rgb.device)

        rgb = self._pixel_effect(rgb, kernel_size, pixel_size)
        edge_mask = self._edge_detect(rgb, edge_thresh)
        return torch.where(torch.gt(edge_mask, 0.5), torch.zeros_like(rgb), rgb)
