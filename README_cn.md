# ![LOGO](images/doc/favicon-original.png) Photo2Pixel

---
[English](./README.md) | 简体中文

[在线工具](https://photo2pixel.co/cn) |
[使用说明](#使用说明)

photo2pixel是一个图片转像素风的算法。提供 [图片转像素风在线工具 photo2pixel.co](https://photo2pixel.co/cn)
以方便使用。本算法可以选择不同的像素大小，边缘强度等风格选项组合以达到最好的效果。

<img src="images/doc/mountain_8bit_style_pixel.png" style="max-width: 850px" alt="mountain 8bit style pixel art"/>
<img src="images/doc/holy_temple_8bit_style_pixel.png" style="max-width: 850px" alt="holy temple 8bit style pixel art">

## 运行环境
- python3
- pytorch
- pillow

## 使用说明
---
photo2pixel基于pytorch框架实现, 以命令行的方式在本地运行:
```bash
# use default param
python convert.py --input ./images/example_input_mountain.jpg

# or use custom param
python convert.py --kernel_size 12 --pixel_size 12 --edge_thresh 128
```

| 参数名称        |            说明             | 取值范围  | 默认值 |
|-------------|:-------------------------:|:-----:|:---:|
| input       |          输入图片路径           |   /   | ./images/example_input_mountain.jpg  |
| output      |          输出图片路径           |   /   | ./result.png  |
| kernel_size | 控制图像颜色的光滑连贯,kernel越大颜色越平滑 |  无限制  | 10  |
| pixel_size  |        像素画中每个像素的大小        |  无限制  | 16  |
| edge_thresh |   像素画中边缘用黑线加强,阈值越低黑线越多    | 0~255 | 100 |