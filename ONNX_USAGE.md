# Unified ONNX Usage

This project can export one ONNX file that accepts all style config values at runtime:

- `param_kernel_size`
- `param_pixel_size`
- `param_edge_thresh`

That means you do not need one ONNX file per config combination.

## 1. Install Dependencies

Use your Python environment, then install the project requirements:

```bash
pip install -r requirements.txt
```

If you use the existing local virtualenv in this workspace, run commands with:

```bash
.venv/bin/python
```

## 2. Export The Unified ONNX File

From the repo root:

```bash
python export_onnx.py --output ./photo2pixel.onnx
```

With the existing local virtualenv:

```bash
.venv/bin/python export_onnx.py --output ./photo2pixel.onnx
```

The exported model has these inputs:

| Input | Type | Shape | Description |
| --- | --- | --- | --- |
| `rgb` | `float32` | `[1, 3, height, width]` | RGB image tensor in `0..255` range |
| `param_kernel_size` | `int64` scalar | `[]` | Larger values smooth color more |
| `param_pixel_size` | `int64` scalar | `[]` | Pixel block size |
| `param_edge_thresh` | `float32` scalar | `[]` | Lower values create more black edge lines |

The output is:

| Output | Type | Shape |
| --- | --- | --- |
| `output` | `float32` | `[1, 3, output_height, output_width]` |

## 3. Convert An Image With The Unified ONNX File

Default example:

```bash
python convert_onnx.py \
  --model ./photo2pixel.onnx \
  --input ./images/example_input_mountain.jpg \
  --output ./result.png
```

Custom config:

```bash
python convert_onnx.py \
  --model ./photo2pixel.onnx \
  --input ./images/example_input_mountain.jpg \
  --output ./result_k12_p12_e128.png \
  --kernel_size 12 \
  --pixel_size 12 \
  --edge_thresh 128
```

Run another config using the same ONNX file:

```bash
python convert_onnx.py \
  --model ./photo2pixel.onnx \
  --input ./images/example_input_mountain.jpg \
  --output ./result_k25_p8_e80.png \
  --kernel_size 25 \
  --pixel_size 8 \
  --edge_thresh 80
```

## 4. Use ONNX Runtime Directly

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

img = Image.open("./images/example_input_mountain.jpg").convert("RGB")
rgb = np.array(img).astype(np.float32)
rgb = np.transpose(rgb, [2, 0, 1])[np.newaxis, :, :, :]

session = ort.InferenceSession("./photo2pixel.onnx", providers=["CPUExecutionProvider"])
output = session.run(
    ["output"],
    {
        "rgb": rgb,
        "param_kernel_size": np.array(12, dtype=np.int64),
        "param_pixel_size": np.array(12, dtype=np.int64),
        "param_edge_thresh": np.array(128, dtype=np.float32),
    },
)[0]

result = output[0].transpose([1, 2, 0])
result = np.clip(result, 0, 255).astype(np.uint8)
Image.fromarray(result).save("./result.png")
```

## Notes

- `kernel_size` and `pixel_size` must be positive.
- `edge_thresh` is normally used in the `0..255` range.
- The ONNX export uses dynamic image height and width, so the same model can process different image sizes.
- Output dimensions may differ slightly from input dimensions depending on `pixel_size`, matching the original PyTorch algorithm behavior.
