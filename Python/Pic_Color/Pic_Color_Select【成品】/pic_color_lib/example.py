#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例代码：展示如何使用pic_color_lib库处理图像

该示例展示了如何使用PicProcessor类处理图像并生成电子墨水屏幕数据包
"""

import os
import cv2
import numpy as np
from pic_color_lib import PicProcessor

def main():
    # 设置输入和输出目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'photos')
    output_dir = os.path.join(current_dir, 'output')
    
    # 初始化图像处理器
    processor = PicProcessor(n_colors=10, input_dir=input_dir, output_dir=output_dir)
    
    # 示例1：处理单个图像文件
    image_path = '1.jpg'  # 相对于input_dir的路径
    processed_image, packets, image_save_path, packets_save_path = processor.process_file(image_path)
    
    print(f"处理完成：{image_path}")
    print(f"处理后的图像保存在：{image_save_path}")
    print(f"数据包保存在：{packets_save_path}")
    print(f"生成了 {len(packets)} 个数据包")
    
    # 示例2：批量处理多个图像文件
    image_files = ['2.jpg', '3.jpg', '4.jpg']
    results = []
    
    for image_file in image_files:
        print(f"\n正在处理：{image_file}")
        result = processor.process_file(image_file)
        results.append(result)
        print(f"处理后的图像保存在：{result[2]}")
        print(f"数据包保存在：{result[3]}")
    
    # 示例3：自定义处理参数
    print("\n使用自定义参数处理图像...")
    # 加载图像
    custom_image_path = os.path.join(input_dir, 'photo0.jpg')
    image = cv2.imread(custom_image_path)
    
    if image is not None:
        # 自定义处理参数
        processed_image = processor.process_image(image, resize_to=150, crop_center=False)
        
        # 保存处理后的图像
        save_path = processor.save_processed_image(processed_image, 'custom_processed.png')
        print(f"自定义处理的图像保存在：{save_path}")
    else:
        print(f"无法读取图像：{custom_image_path}")

if __name__ == "__main__":
    main()