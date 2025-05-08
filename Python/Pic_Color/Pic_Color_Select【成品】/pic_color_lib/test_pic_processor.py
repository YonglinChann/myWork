#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本：验证PicProcessor类的功能

该脚本测试pic_color_lib库中PicProcessor类的各项功能，
并展示了更多使用示例和高级用法。
"""

import os
import cv2
import numpy as np
from pic_color_lib import PicProcessor

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'photos')
    output_dir = os.path.join(current_dir, 'output', 'test_results')
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化处理器
    processor = PicProcessor(n_colors=10, input_dir=input_dir, output_dir=output_dir)
    
    # 测试加载图像
    image_path = os.path.join(input_dir, '1.jpg')
    image = processor.load_image(image_path)
    print(f"成功加载图像: {image_path}")
    print(f"图像尺寸: {image.shape}")
    
    # 测试处理图像
    processed_image = processor.process_image(image)
    print(f"成功处理图像")
    print(f"处理后图像尺寸: {processed_image.shape}")
    
    # 测试生成数据包
    packets = processor.image_to_eink_packets(processed_image)
    print(f"成功生成数据包，共 {len(packets)} 个")
    
    # 测试保存图像
    image_save_path = processor.save_processed_image(processed_image, 'test_basic.png')
    print(f"成功保存处理后的图像: {image_save_path}")
    
    # 测试保存数据包
    packets_save_path = processor.save_eink_packets(packets, 'test_basic.txt')
    print(f"成功保存数据包: {packets_save_path}")

def test_custom_parameters():
    """测试自定义参数"""
    print("\n=== 测试自定义参数 ===")
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'photos')
    output_dir = os.path.join(current_dir, 'output', 'test_results')
    
    # 初始化处理器，使用不同的颜色数量
    processor = PicProcessor(n_colors=5, input_dir=input_dir, output_dir=output_dir)
    
    # 加载图像
    image_path = os.path.join(input_dir, '2.jpg')
    image = processor.load_image(image_path)
    
    # 使用不同的缩放尺寸和裁剪方式
    processed_image_1 = processor.process_image(image, resize_to=150, crop_center=True)
    processed_image_2 = processor.process_image(image, resize_to=150, crop_center=False)
    
    # 保存处理后的图像
    processor.save_processed_image(processed_image_1, 'test_custom_center.png')
    processor.save_processed_image(processed_image_2, 'test_custom_corner.png')
    
    print(f"成功使用自定义参数处理图像")
    print(f"中心裁剪图像尺寸: {processed_image_1.shape}")
    print(f"角落裁剪图像尺寸: {processed_image_2.shape}")

def test_batch_processing():
    """测试批量处理"""
    print("\n=== 测试批量处理 ===")
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'photos')
    output_dir = os.path.join(current_dir, 'output', 'test_results')
    
    # 初始化处理器
    processor = PicProcessor(n_colors=10, input_dir=input_dir, output_dir=output_dir)
    
    # 批量处理多个图像
    image_files = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
    
    for i, image_file in enumerate(image_files):
        print(f"处理图像 {i+1}/{len(image_files)}: {image_file}")
        processed_image, packets, image_save_path, packets_save_path = processor.process_file(
            image_file, 
            resize_to=200, 
            crop_center=True,
            save_image=True,
            save_packets=True
        )
        print(f"  - 图像保存在: {image_save_path}")
        print(f"  - 数据包保存在: {packets_save_path}")

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_dir, 'photos')
    output_dir = os.path.join(current_dir, 'output', 'test_results')
    
    # 初始化处理器
    processor = PicProcessor(n_colors=10, input_dir=input_dir, output_dir=output_dir)
    
    # 测试不存在的文件
    try:
        processor.load_image('non_existent_file.jpg')
    except FileNotFoundError as e:
        print(f"成功捕获文件不存在错误: {e}")
    
    # 测试图像尺寸过小
    try:
        # 创建一个小图像
        small_image = np.zeros((100, 100, 3), dtype=np.uint8)
        processor.process_image(small_image, resize_to=200)
    except ValueError as e:
        print(f"成功捕获图像尺寸过小错误: {e}")

def main():
    print("开始测试 PicProcessor 类...")
    
    # 运行测试
    test_basic_functionality()
    test_custom_parameters()
    test_batch_processing()
    test_error_handling()
    
    print("\n所有测试完成！")

if __name__ == "__main__":
    main()