# Pic Color Processor

一个用于处理图像并为特定四色（黑、白、红、黄）墨水屏生成数据包的 Python 库。

## 功能

- 读取图像文件 (jpg, png, bmp 等)。
- 使用 Jarvis-Judice-Ninke 抖动算法将图像颜色量化为目标四色。
- 将图像缩放至宽度或高度为 200 像素。
- 根据指定模式（中心裁剪或自定义坐标）将图像裁剪为 200x200 像素。
- 生成适用于目标墨水屏的十六进制数据包。

## 安装

```bash
# 从源代码安装
python setup.py install
# 或者使用 pip 安装 (如果发布到 PyPI)
# pip install pic_color_processor
```

## 使用方法

```python
from pic_color_processor import process_image
import os

# 图像路径
image_path = 'path/to/your/image.png'
# 输出目录 (可选, 默认在库安装目录下创建 output 文件夹)
output_base = './my_output'

# 检查文件是否存在
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
else:
    try:
        # --- 中心裁剪 ---
        print("Processing with center crop...")
        result_center = process_image(image_path, crop_mode=0, output_dir_base=output_base)
        if result_center:
            print(f"Center crop finished. Output image: {result_center['final_image_path']}")
            print(f"Generated {len(result_center['hex_data_list'])} hex data packets.")
            # print(result_center['hex_data_list'])

        # --- 自定义裁剪 (示例: 从左侧裁剪 x=10) ---
        # 注意: 需根据图片缩放后的实际尺寸调整 custom_coord
        # 如果缩放后宽度 > 200, custom_coord 是 x 坐标 (从左侧)
        # 如果缩放后高度 > 200, custom_coord 是 y 坐标 (从底部)
        print("\nProcessing with custom crop (x=10 from left, assuming width > 200 after scale)...")
        custom_coordinate = 10
        result_custom = process_image(image_path, crop_mode=1, custom_coord=custom_coordinate, output_dir_base=output_base)
        if result_custom:
            print(f"Custom crop finished. Output image: {result_custom['final_image_path']}")
            print(f"Generated {len(result_custom['hex_data_list'])} hex data packets.")
            # print(result_custom['hex_data_list'])

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

```

## 依赖

- Python 3.6+
- NumPy
- OpenCV-Python

详细请见 `requirements.txt`。

## 贡献

欢迎提出问题和合并请求。

## 许可证

MIT License (请根据实际情况修改 `setup.py` 中的许可证信息)