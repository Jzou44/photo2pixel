import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image


def image_to_array(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32)
    return np.transpose(img_np, axes=[2, 0, 1])[np.newaxis, :, :, :]


def array_to_image(img_np):
    img_np = img_np[0].transpose([1, 2, 0])
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np)


def convert():
    parser = argparse.ArgumentParser(description="convert photo to pixel art with one configurable ONNX file")
    parser.add_argument("--model", type=str, default="./photo2pixel.onnx", help="ONNX model path")
    parser.add_argument("--input", type=str, default="./images/example_input_mountain.jpg", help="input image path")
    parser.add_argument("--output", type=str, default="./result.png", help="output image path")
    parser.add_argument("-k", "--kernel_size", type=int, default=10, help="larger kernel size means smooth color transition")
    parser.add_argument("-p", "--pixel_size", type=int, default=16, help="individual pixel size")
    parser.add_argument("-e", "--edge_thresh", type=float, default=100, help="lower edge threshold means more black line in edge region")
    args = parser.parse_args()

    if args.kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if args.pixel_size < 1:
        raise ValueError("pixel_size must be positive")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    output = session.run(
        ["output"],
        {
            "rgb": image_to_array(args.input),
            "param_kernel_size": np.array(args.kernel_size, dtype=np.int64),
            "param_pixel_size": np.array(args.pixel_size, dtype=np.int64),
            "param_edge_thresh": np.array(args.edge_thresh, dtype=np.float32),
        },
    )[0]
    array_to_image(output).save(args.output)


if __name__ == "__main__":
    convert()
