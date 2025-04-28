# Pic Color Processor

## 简介

Pic Color Processor 是一个用于处理图片并为电子墨水屏（e-ink display）生成数据的 Python 库，支持抖动算法和多种图像处理方式，适用于电子纸显示、图像二值化等场景。

## 安装方法

推荐使用 pip 进行安装：

```bash
pip install numpy opencv-python scikit-learn
```

或直接克隆本仓库后手动安装依赖：

```bash
git clone https://gitee.com/your_username/your_repo.git
cd your_repo
pip install -r requirements.txt
```

## 依赖说明

- Python >= 3.6
- numpy
- opencv-python
- scikit-learn

详细依赖请参考 `pyproject.toml` 或 `setup.py`。

## 用法示例

```python
import cv2
import numpy as np
# from pic_color_processor import process_image # 假设主处理函数为 process_image

def process_image(image_path, output_path):
    # 示例：读取图片并转为灰度
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 示例：简单二值化
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(output_path, binary)

# 使用示例
process_image('input.jpg', 'output.png')
```

## API 说明

### process_image(image_path, output_path)
- `image_path`：输入图片路径（str）
- `output_path`：输出图片路径（str）
- 功能：读取图片，进行灰度转换和二值化处理，保存结果。

> 具体 API 请根据实际代码实现补充详细参数和返回值说明。

## 许可证

MIT License

如需商业授权或有其他问题，请联系作者。