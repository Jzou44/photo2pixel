# ![LOGO](images/doc/favicon-original.png) Photo2Pixel

---
English | [简体中文](./README_cn.md)

[Online Demo Tool](https://photo2pixel.co) |
[Tutorial](#Tutorial)

photo2pixel is an algorithm converting photo into pixel art. There is an [online converter photo2pixel.co](https://photo2pixel.co)
. you can choose different combination of pixel size and edge threshold to get the best result.

<img src="images/doc/mountain_8bit_style_pixel.png" style="max-width: 850px" alt="mountain 8bit style pixel art"/>
<img src="images/doc/holy_temple_8bit_style_pixel.png" style="max-width: 850px" alt="holy temple 8bit style pixel art">

## Tutorial
---
photo2pixel is implemented with Pytorch, you can run it with command as follow:
```bash
# use default param
python convert.py --input ./images/example_input_mountain.jpg

# or use custom param
python convert.py --kernel_size 12 --pixel_size 12 --edge_thresh 128
```

| Parameter   |                                Description                                |   Range   |               Default               |
|-------------|:-------------------------------------------------------------------------:|:---------:|:-----------------------------------:|
| input       |                             input image path                              |     /     | ./images/example_input_mountain.jpg |
| output      |                             output image path                             |     /     |            ./result.png             |
| kernel_size |             larger kernel size means smooth color transition              | unlimited |                 10                  |
| pixel_size  |                           individual pixel size                           |    unlimited    |                 16                  |
| edge_thresh | the black line in edge region, lower edge threshold means more black line |   0~255   |                 100                 |