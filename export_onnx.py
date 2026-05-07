import argparse
import inspect

import torch

from models.module_photo2pixel_onnx import ConfigurablePhoto2PixelOnnxModel


def export():
    parser = argparse.ArgumentParser(description="export one ONNX model with runtime photo2pixel config inputs")
    parser.add_argument("--output", type=str, default="./photo2pixel.onnx", help="output ONNX file path")
    parser.add_argument("--height", type=int, default=256, help="sample input height for export")
    parser.add_argument("--width", type=int, default=256, help="sample input width for export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model = ConfigurablePhoto2PixelOnnxModel()
    model.eval()

    rgb = torch.zeros([1, 3, args.height, args.width], dtype=torch.float32)
    kernel_size = torch.tensor(10, dtype=torch.int64)
    pixel_size = torch.tensor(16, dtype=torch.int64)
    edge_thresh = torch.tensor(100.0, dtype=torch.float32)

    export_kwargs = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    torch.onnx.export(
        model,
        (rgb, kernel_size, pixel_size, edge_thresh),
        args.output,
        input_names=["rgb", "param_kernel_size", "param_pixel_size", "param_edge_thresh"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={
            "rgb": {2: "height", 3: "width"},
            "output": {2: "output_height", 3: "output_width"},
        },
        **export_kwargs,
    )
    print(f"exported {args.output}")


if __name__ == "__main__":
    export()
