# MetricMis
Code will be available soon.

## Installation

Clone the [Depth Anything v2 model](https://github.com/DepthAnything/Depth-Anything-V2)

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2.git

cp -r Depth-Anything-V2/depth_anything_v2 .

rm -r Depth-Anything-V2
```

please install pytorch:

CPU:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

GPU:
```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

then:
```bash
uv sync
```


## Usage

```bash
test_image --checkpoint "ckpt_path" --image "img_path"
```