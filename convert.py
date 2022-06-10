import torch
from PIL import Image
import argparse

from models.module_photo2pixel import Photo2PixelModel
from utils import img_common_util


def convert():
    parser = argparse.ArgumentParser(description='algorithm converting photo to pixel art')
    parser.add_argument('--input', type=str, default="./images/example_input_mountain.jpg", help='input image path')
    parser.add_argument('--output', type=str, default="./result.png", help='output image path')
    parser.add_argument('-k', '--kernel_size', type=int, default=10, help='larger kernel size means smooth color transition')
    parser.add_argument('-p', '--pixel_size', type=int, default=16, help='individual pixel size')
    parser.add_argument('-e', '--edge_thresh', type=int, default=100, help='lower edge threshold means more black line in edge region')
    args = parser.parse_args()

    img_input = Image.open(args.input)
    img_pt_input = img_common_util.convert_image_to_tensor(img_input)

    model = Photo2PixelModel()
    model.eval()
    with torch.no_grad():
        img_pt_output = model(
            img_pt_input,
            param_kernel_size=args.kernel_size,
            param_pixel_size=args.pixel_size,
            param_edge_thresh=args.edge_thresh
        )
    img_output = img_common_util.convert_tensor_to_image(img_pt_output)
    img_output.save(args.output)


if __name__ == '__main__':
    convert()
