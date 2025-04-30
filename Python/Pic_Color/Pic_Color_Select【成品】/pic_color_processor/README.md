# Pic Color Processor
## 简介

Pic Color Processor 是一个用于处理图片并为电子墨水屏（e-ink display）生成数据的 Python 库，支持抖动算法和多种图像处理方式，适用于电子纸显示、图像二值化等场景。

## 功能特点

- **图像缩放和裁剪**：保持宽高比的图像缩放和中心裁剪
- **颜色聚类分析**：使用K-means算法进行颜色聚类
- **颜色映射**：将图像颜色映射到目标颜色（黑、白、红、黄）
- **抖动算法处理**：使用Jarvis-Judice-Ninke抖动算法优化图像显示效果
- **墨水屏数据包生成**：生成适用于电子墨水屏的数据包

## 安装方法

```bash
# 从PyPI安装
pip install pic-color-processor

# 或从源码安装
git clone https://gitee.com/your_username/pic_color_processor.git
cd pic_color_processor
pip install -e .
```

## 模块结构

- **PicColorProcessor**：主处理器类，整合所有功能
- **ImageProcessor**：处理图像缩放和裁剪
- **ColorMapper**：处理颜色映射
- **DitherProcessor**：实现抖动算法
- **EInkPacketGenerator**：生成墨水屏数据包

## 用法示例

### 基本用法

```python
from pic_color_processor import PicColorProcessor
import cv2
import numpy as np

# 读取图像
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 创建处理器实例
processor = PicColorProcessor(n_colors=10)  # 设置颜色聚类数量

# 处理图像
processed_image = processor.process_image(image, resize_to=200, crop_center=True)

# 保存处理后的图像
cv2.imwrite('output.png', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

# 生成墨水屏数据包
packets = processor.image_to_eink_packets(processed_image)

# 打印第一个数据包信息
print(f"生成了{len(packets)}个数据包")
print(f"第一个数据包长度: {len(packets[0])}字节")
```

### 高级用法

```python
from pic_color_processor import ImageProcessor, DitherProcessor, EInkPacketGenerator
import cv2
import numpy as np

# 读取图像
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 单独使用各个处理器
img_processor = ImageProcessor()
scaled_image = img_processor.resize_image(image, resize_to=200)

# 自定义抖动处理
dither_processor = DitherProcessor()
dithered_image = dither_processor.apply_dithering(scaled_image)

# 裁剪图像
cropped_image = img_processor.crop_image(dithered_image, target_size=200, center=True)

# 生成墨水屏数据包
eink_generator = EInkPacketGenerator()
packets = eink_generator.generate_packets(cropped_image)

# 将数据包保存为二进制文件
with open('eink_data.bin', 'wb') as f:
    for packet in packets:
        f.write(bytes(packet))
```

## 参数说明

### PicColorProcessor
- **n_colors**: 颜色聚类的数量，默认为10

### process_image 方法
- **image**: 输入图像，应为RGB格式的NumPy数组
- **resize_to**: 目标尺寸，默认为200像素
- **crop_center**: 是否从中心裁剪，默认为True

### image_to_eink_packets 方法
- **image**: 处理后的图像，应为200x200像素的RGB格式NumPy数组

## 注意事项

- 输入图像尺寸应大于等于目标尺寸（默认200x200像素）
- 输入图像应为RGB格式，如果使用OpenCV读取，需要使用`cv2.cvtColor`转换
- 生成的墨水屏数据包适用于特定型号的墨水屏，可能需要根据实际硬件调整

## 许可证

MIT License