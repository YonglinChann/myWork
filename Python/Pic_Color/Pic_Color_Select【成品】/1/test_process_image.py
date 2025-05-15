#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本：用于测试process_image_combined函数
"""

import os
import sys
from picColor_scale_first_all import process_image_combined

def test_process_image():
    """测试process_image_combined函数的功能"""
    
    # 测试参数
    image_name = "2024-00"
    # 获取当前目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "pic")
    
    # 确保测试目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"测试参数:")
    print(f"- 图像名称: {image_name}")
    print(f"- 目标目录: {target_dir}")
    
    # 检查输入图像是否存在
    input_image = f"{target_dir}/{image_name}.jpg"
    if not os.path.exists(input_image):
        print(f"警告: 输入图像不存在: {input_image}")
        print(f"请确保图像文件 {image_name}.jpg 放置在 {target_dir} 目录下")
        return
    
    # 调用函数并计时
    import time
    start_time = time.time()
    
    print(f"\n开始测试 process_image_combined 函数...")
    result = process_image_combined(image_name, target_dir)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 输出结果
    print(f"\n测试结果:")
    print(f"- 耗时: {elapsed_time:.2f} 秒")
    print(f"- 结果类型: {type(result)}")
    print(f"- 结果长度: {len(result)}")
    
    # 检查结果内容
    if result:
        print("\n结果样例 (第一个条目):")
        first_item = result[0]
        for key, value in first_item.items():
            if key == 'data':
                print(f"- {key}: [数据包列表，包含 {len(value)} 个数据包]")
                if value:
                    print(f"  - 第一个数据包前20个字符: {value[0][:20]}...")
            else:
                print(f"- {key}: {value}")
    
    # 检查生成的文件
    output_files = {
        'default_image': f"{target_dir}/{image_name}/{image_name}_default.png",
        'T_image': f"{target_dir}/{image_name}/{image_name}_T.png",
        'C_image': f"{target_dir}/{image_name}/{image_name}_C.png",
        'default_data': f"{target_dir}/{image_name}/default_data.txt",
        'T_data': f"{target_dir}/{image_name}/T_data.txt",
        'C_data': f"{target_dir}/{image_name}/C_data.txt"
    }
    
    print("\n生成的文件检查:")
    for file_type, file_path in output_files.items():
        status = "✓ 存在" if os.path.exists(file_path) else "✗ 不存在"
        size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        print(f"- {file_type}: {status} (大小: {size} 字节)")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_process_image() 