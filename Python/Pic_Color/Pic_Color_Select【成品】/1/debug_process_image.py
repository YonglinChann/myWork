#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用调试器调试process_image_combined函数
"""

import os
import sys
import pdb  # Python调试器
from picColor_scale_first_all import process_image_combined

def debug_process_image():
    """使用调试器调试process_image_combined函数"""
    
    # 测试参数
    image_name = "2024-00"
    # 获取当前目录的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "pic")
    
    # 确保测试目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"调试参数:")
    print(f"- 图像名称: {image_name}")
    print(f"- 目标目录: {target_dir}")
    
    # 检查输入图像是否存在
    input_image = f"{target_dir}/{image_name}.jpg"
    if not os.path.exists(input_image):
        print(f"警告: 输入图像不存在: {input_image}")
        print(f"请确保图像文件 {image_name}.jpg 放置在 {target_dir} 目录下")
        return
    
    # 设置断点并调用函数
    print("\n开始调试 process_image_combined 函数...")
    print("进入调试模式，使用以下命令:")
    print("- n: 执行下一行")
    print("- s: 步入函数")
    print("- c: 继续执行直到下一个断点")
    print("- p 变量名: 打印变量值")
    print("- q: 退出调试器")
    
    # 设置断点
    pdb.set_trace()
    
    # 调用被调试的函数
    result = process_image_combined(image_name, target_dir)
    
    # 在调试器中检查结果
    print("\n函数执行完毕，可以在调试器中检查结果变量")
    pdb.set_trace()
    
    return result

if __name__ == "__main__":
    debug_process_image() 