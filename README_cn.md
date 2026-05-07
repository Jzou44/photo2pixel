# ![LOGO](images/doc/favicon-original.png) Photo2Pixel

---
[English](./README.md) | 简体中文

[在线工具](https://coding.tools/photo2pixel) |
[Colab](https://colab.research.google.com/drive/108np4teybhBXHKbPMZZ1fykDuUeF2aw8?usp=sharing) |
[使用说明](#使用说明)

photo2pixel是一个图片转像素风的算法。提供 [图片转像素风在线工具 coding.tools/photo2pixel](https://coding.tools/photo2pixel)
以方便使用。本算法可以选择不同的像素大小，边缘强度等风格选项组合以达到最好的效果。

<img src="images/doc/mountain_8bit_style_pixel.png" style="max-width: 850px" alt="mountain 8bit style pixel art"/>
<img src="images/doc/holy_temple_8bit_style_pixel.png" style="max-width: 850px" alt="holy temple 8bit style pixel art">

## 运行环境
- python3
- pytorch
- pillow
- onnx 和 onnxruntime（用于导出和运行单个 ONNX 文件）

## 使用说明
---
photo2pixel基于pytorch框架实现, 最简洁的方式是在[Colab](https://colab.research.google.com/drive/108np4teybhBXHKbPMZZ1fykDuUeF2aw8?usp=sharing)中运行，或者以命令行的方式在本地运行:
```bash
# use default param
python convert.py --input ./images/example_input_mountain.jpg

# or use custom param
python convert.py --kernel_size 12 --pixel_size 12 --edge_thresh 128
```

导出一个可配置的 ONNX 文件后，可以用同一个文件运行不同参数:
```bash
python export_onnx.py --output ./photo2pixel.onnx
python convert_onnx.py --model ./photo2pixel.onnx --kernel_size 12 --pixel_size 12 --edge_thresh 128
python convert_onnx.py --model ./photo2pixel.onnx --kernel_size 25 --pixel_size 8 --edge_thresh 80
```

导出的 ONNX 文件将 `kernel_size`、`pixel_size` 和 `edge_thresh` 作为运行时输入，因此同一个文件可以复用在不同参数组合上。

| 参数名称        |            说明             | 取值范围  | 默认值 |
|-------------|:-------------------------:|:-----:|:---:|
| input       |          输入图片路径           |   /   | ./images/example_input_mountain.jpg  |
| output      |          输出图片路径           |   /   | ./result.png  |
| kernel_size | 控制图像颜色的光滑连贯,kernel越大颜色越平滑 |  无限制  | 10  |
| pixel_size  |        像素画中每个像素的大小        |  无限制  | 16  |
| edge_thresh |   像素画中边缘用黑线加强,阈值越低黑线越多    | 0~255 | 100 |

updated by openclaw at 20260312
