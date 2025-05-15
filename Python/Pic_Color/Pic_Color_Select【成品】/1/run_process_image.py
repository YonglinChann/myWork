#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
命令行工具：调用process_image_combined函数处理图像
用法: python run_process_image.py <image_name> [target_dir]
"""

import os
import sys
import argparse
import time
from picColor_scale_first_all import process_image_combined

def main():
    """处理命令行参数并调用process_image_combined函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理图像为墨水屏格式')
    parser.add_argument('image_name', help='图像名称（不包含扩展名）')
    parser.add_argument('--target-dir', '-t', default='./pic', 
                        help='目标目录路径 (默认: ./pic)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细输出')
    
    args = parser.parse_args()
    
    # 获取参数
    image_name = args.image_name
    target_dir = args.target_dir
    verbose = args.verbose
    
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    if verbose:
        print(f"处理参数:")
        print(f"- 图像名称: {image_name}")
        print(f"- 目标目录: {target_dir}")
    
    # 检查输入图像是否存在
    input_image = f"{target_dir}/{image_name}.jpg"
    if not os.path.exists(input_image):
        print(f"错误: 输入图像不存在: {input_image}")
        print(f"请确保图像文件 {image_name}.jpg 已放置在 {target_dir} 目录下")
        return 1
    
    # 计时并调用函数
    start_time = time.time()
    
    if verbose:
        print(f"\n开始处理图像...")
    
    result = process_image_combined(image_name, target_dir)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 检查结果
    if not result:
        print(f"处理失败: 没有生成有效的结果数据")
        return 1
    
    # 输出结果信息
    print(f"\n处理成功!")
    print(f"- 处理时间: {elapsed_time:.2f} 秒")
    print(f"- 生成的结果数量: {len(result)}")
    
    if verbose:
        # 检查生成的文件
        output_files = {
            'default_image': f"{target_dir}/{image_name}/{image_name}_default.png",
            'T_image': f"{target_dir}/{image_name}/{image_name}_T.png",
            'C_image': f"{target_dir}/{image_name}/{image_name}_C.png",
            'default_data': f"{target_dir}/{image_name}/default_data.txt",
            'T_data': f"{target_dir}/{image_name}/T_data.txt",
            'C_data': f"{target_dir}/{image_name}/C_data.txt"
        }
        
        print("\n生成的文件:")
        for file_type, file_path in output_files.items():
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                size_str = f"{size} 字节"
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.2f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.2f} KB"
                print(f"- {file_type}: {file_path} ({size_str})")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 