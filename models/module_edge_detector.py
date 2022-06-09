import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image


class EdgeDetectorModule(nn.Module):
    def __init__(self):
        super(EdgeDetectorModule, self).__init__()

        self.pad = nn.ReflectionPad2d(padding=(1, 1, 1, 1))

        kernel_sobel_h = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_h = torch.from_numpy(kernel_sobel_h).reshape([1, 1, 3, 3]).repeat([3, 1, 1, 1])
        self.conv_h = nn.Conv2d(3, 3, kernel_size=3, padding=0, groups=3, bias=False)
        self.conv_h.weight = nn.Parameter(kernel_sobel_h)

        kernel_sobel_v = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float32).reshape([1, 1, 3, 3])
        kernel_sobel_v = torch.from_numpy(kernel_sobel_v).reshape([1, 1, 3, 3]).repeat([3, 1, 1, 1])
        self.conv_v = nn.Conv2d(3, 3, kernel_size=3, padding=0, groups=3, bias=False)
        self.conv_v.weight = nn.Parameter(kernel_sobel_v)

    def forward(self, rgb, param_edge_thresh, param_edge_dilate):
        """
        :param rgb: [1, c(3), H, W]
        :param param_edge_thresh: int
        :param param_edge_dilate: odd number
        :return: [1,c(1),H,W]
        """

        rgb = self.pad(rgb)
        edge_h = self.conv_h(rgb)
        edge_w = self.conv_v(rgb)
        edge = torch.stack([torch.abs(edge_h), torch.abs(edge_w)], dim=1)
        edge = torch.max(edge, dim=1)[0]

        edge = torch.mean(edge, dim=1, keepdim=True)
        edge = torch.gt(edge, param_edge_thresh).float()

        edge = F.max_pool2d(edge, kernel_size=param_edge_dilate, stride=1, padding=param_edge_dilate // 2)
        return edge


def test():
    rgb = np.array(Image.open("../images/example_input_mountain.jpg").convert("RGB")).astype(np.float32)
    rgb = torch.from_numpy(rgb).permute([2, 0, 1]).unsqueeze(dim=0)

    net = EdgeDetectorModule()
    edge_mask = net(rgb, param_edge_thresh=128, param_edge_dilate=3)
    print(edge_mask.shape)

    edge_mask = 255 * edge_mask
    edge_mask = edge_mask[0, ...].permute(1, 2, 0).repeat([1, 1, 3])
    edge_mask = edge_mask.cpu().numpy().astype(np.uint8)
    Image.fromarray(edge_mask).save("test_result_edge.png")


if __name__ == '__main__':
    test()
