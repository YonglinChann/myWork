# Pic Color 图像处理库

这是一个用于处理图像并生成适用于电子墨水屏幕的数据包的Python库。该库提供了图像处理功能，包括缩放、颜色聚类、抖动处理和数据包生成。

## 功能特点

- 图像缩放和裁剪
- 颜色聚类（使用KMeans算法）
- 颜色映射到指定目标颜色
- 抖动处理（使用Jarvis, Judice, and Ninke算法）
- 生成适用于电子墨水屏幕的数据包
- 支持自定义输入输出路径

## 安装

```bash
pip install -e .
```

## 使用方法

### 基本使用

```python
from pic_color_lib import PicProcessor

# 初始化处理器
processor = PicProcessor()

# 加载图像
image = processor.load_image('path/to/image.jpg')

# 处理图像
processed_image = processor.process_image(image)

# 生成数据包
packets = processor.image_to_eink_packets(processed_image)

# 保存处理后的图像
processor.save_processed_image(processed_image, 'processed.png')

# 保存数据包
processor.save_eink_packets(packets, 'packets.txt')
```

### 使用自定义路径

```python
# 指定输入和输出目录
processor = PicProcessor(input_dir='input_images', output_dir='output_results')

# 处理单个文件
processed_image, packets, image_path, packets_path = processor.process_file('image.jpg')
```

### 批量处理图像

```python
# 指定输入和输出目录
processor = PicProcessor(input_dir='input_images', output_dir='output_results')

# 批量处理多个图像
image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg']
for image_file in image_files:
    processor.process_file(image_file)
```

## 示例

查看 `example.py` 文件获取完整的使用示例。

## 参数说明

### PicProcessor 类

- `n_colors`: 颜色聚类的数量，默认为10
- `input_dir`: 输入图像的目录路径，默认为None
- `output_dir`: 输出结果的目录路径，默认为None

### process_image 方法

- `image`: 输入图像数据
- `resize_to`: 缩放尺寸，默认为200
- `crop_center`: 是否从中心裁剪，默认为True

### process_file 方法

- `image_path`: 图像文件路径
- `resize_to`: 缩放尺寸，默认为200
- `crop_center`: 是否从中心裁剪，默认为True
- `save_image`: 是否保存处理后的图像，默认为True
- `save_packets`: 是否保存数据包，默认为True

## 许可证

MIT