import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class PixelEffectModule(nn.Module):
    def __init__(self):
        super(PixelEffectModule, self).__init__()

    @staticmethod
    def _gather_bins(data, indices):
        return torch.gather(data, dim=0, index=indices.unsqueeze(0)).squeeze(0)

    def forward(self, rgb, param_num_bins, param_kernel_size, param_pixel_size):
        """
        :param rgb:[b(1), c(3), H, W]
        :return: [b(1), c(3), H, W]
        """

        r, g, b = rgb[:, 0:1, :, :], rgb[:, 1:2, :, :], rgb[:, 2:3, :, :]

        intensity_idx = torch.mean(rgb, dim=[0, 1]) / 256. * param_num_bins
        intensity_idx = torch.clamp(intensity_idx.long(), min=0, max=param_num_bins - 1)

        intensity = F.one_hot(intensity_idx, num_classes=param_num_bins)
        intensity = torch.permute(intensity, dims=[2, 0, 1]).unsqueeze(dim=0).to(dtype=rgb.dtype, device=rgb.device)

        weighted = torch.cat([r * intensity, g * intensity, b * intensity, intensity], dim=1)
        kernel_conv = torch.ones(
            [4 * param_num_bins, 1, param_kernel_size, param_kernel_size],
            dtype=rgb.dtype,
            device=rgb.device,
        )
        summed = F.conv2d(
            input=weighted,
            weight=kernel_conv,
            padding=(param_kernel_size - 1) // 2,
            stride=param_pixel_size,
            groups=4 * param_num_bins,
            bias=None,
        )[0, :, :, :]

        r_sum, g_sum, b_sum, intensity = torch.chunk(summed, chunks=4, dim=0)
        intensity_max, intensity_argmax = torch.max(intensity, dim=0)
        intensity_max = torch.clamp(intensity_max, min=1)

        r = self._gather_bins(r_sum, intensity_argmax) / intensity_max
        g = self._gather_bins(g_sum, intensity_argmax) / intensity_max
        b = self._gather_bins(b_sum, intensity_argmax) / intensity_max

        result_rgb = torch.stack([r, g, b], dim=-1)
        result_rgb = torch.permute(result_rgb, dims=[2, 0, 1]).unsqueeze(dim=0)
        result_rgb = F.interpolate(result_rgb, scale_factor=param_pixel_size)

        return result_rgb


def test1():
    img = Image.open("../images/example_input_mountain.jpg").convert("RGB")
    img_np = np.array(img).astype(np.float32)
    img_np = np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]
    img_pt = torch.from_numpy(img_np)

    model = PixelEffectModule()
    model.eval()

    with torch.no_grad():
        result_rgb_pt = model(img_pt, param_num_bins=4, param_kernel_size=11, param_pixel_size=16)
        result_rgb_pt = result_rgb_pt[0, ...].permute(1, 2, 0)

    print("img_pt", img_pt.shape)
    print("result_rgb_pt", result_rgb_pt.shape)

    result_rgb_np = result_rgb_pt.cpu().numpy().astype(np.uint8)
    Image.fromarray(result_rgb_np).save("./test_result_pixel_effect.png")


if __name__ == '__main__':
    test1()
