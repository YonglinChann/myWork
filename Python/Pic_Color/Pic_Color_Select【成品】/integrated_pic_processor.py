# 整合三种不同的图像处理算法，用于处理图像并转换为四色墨水屏格式
# 墨水屏像素定义：
# 黑 00
# 白 01
# 黄 10
# 红 11

import numpy as np
import cv2
from sklearn.cluster import KMeans
import os
import time
from datetime import datetime

# 导入三种不同的处理算法
import sys
import os

# 获取当前文件所在目录的父目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 添加到系统路径
sys.path.append(current_dir)

# 导入处理函数
sys.path.append(os.path.join(current_dir, '1'))

from picColor_scale_first_all_DIY import process_image_for_eink_default, process_image_for_eink_T
from picColor_scale_first_C import process_image_for_eink as process_image_for_eink_C
from picColor_scale_first import process_image_for_eink as process_image_for_eink_standard

def process_image_with_all_algorithms(input_path, output_folder):
    """
    使用三种不同的算法处理同一张图像，并返回处理结果
    
    参数:
        input_path (str): 输入图片路径
        output_folder (str): 输出文件夹路径，用于保存处理后的图像和数据包
        
    返回:
        dict: 包含三种算法处理结果的字典，键为输出图像文件名，值为对应的HEX数据包列表
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取输入文件名（不含扩展名）
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # 生成时间戳，用于区分不同的处理结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 定义三种算法的输出文件路径
    output_paths = {
        "default": {
            "image": os.path.join(output_folder, f"{input_filename}_default_{timestamp}.png"),
            "packets": os.path.join(output_folder, f"{input_filename}_default_{timestamp}_packets.txt")
        },
        "special_C": {
            "image": os.path.join(output_folder, f"{input_filename}_special_C_{timestamp}.png"),
            "packets": os.path.join(output_folder, f"{input_filename}_special_C_{timestamp}_packets.txt")
        },
        "special_T": {
            "image": os.path.join(output_folder, f"{input_filename}_special_T_{timestamp}.png"),
            "packets": os.path.join(output_folder, f"{input_filename}_special_T_{timestamp}_packets.txt")
        }
    }
    
    # 存储处理结果
    results = {}
    
    # 1. 使用默认算法处理图像（不处理特殊颜色）
    print(f"正在使用默认算法处理图像: {input_path}")
    try:
        hex_packets_default = process_image_for_eink_default(
            input_path, 
            output_paths["default"]["image"],
            output_paths["default"]["packets"]
        )
        results[output_paths["default"]["image"]] = hex_packets_default
        print(f"默认算法处理完成，输出图像: {output_paths['default']['image']}")
    except Exception as e:
        print(f"默认算法处理失败: {str(e)}")
        results[output_paths["default"]["image"]] = None
    
    # 2. 使用特殊颜色处理算法C（使用抖动模式处理蓝色、绿色和青色）
    print(f"正在使用特殊颜色算法C处理图像: {input_path}")
    try:
        hex_packets_C = process_image_for_eink_C(
            input_path, 
            output_paths["special_C"]["image"],
            output_paths["special_C"]["packets"]
        )
        results[output_paths["special_C"]["image"]] = hex_packets_C
        print(f"特殊颜色算法C处理完成，输出图像: {output_paths['special_C']['image']}")
    except Exception as e:
        print(f"特殊颜色算法C处理失败: {str(e)}")
        results[output_paths["special_C"]["image"]] = None
    
    # 3. 使用特殊颜色处理算法T（使用2x2抖动矩阵处理蓝色、绿色和青色）
    print(f"正在使用特殊颜色算法T处理图像: {input_path}")
    try:
        hex_packets_T = process_image_for_eink_T(
            input_path, 
            output_paths["special_T"]["image"],
            output_paths["special_T"]["packets"]
        )
        results[output_paths["special_T"]["image"]] = hex_packets_T
        print(f"特殊颜色算法T处理完成，输出图像: {output_paths['special_T']['image']}")
    except Exception as e:
        print(f"特殊颜色算法T处理失败: {str(e)}")
        results[output_paths["special_T"]["image"]] = None
    
    return results

def compare_algorithms(input_path, output_folder):
    """
    比较三种算法处理同一张图像的结果，并生成比较报告
    
    参数:
        input_path (str): 输入图片路径
        output_folder (str): 输出文件夹路径
        
    返回:
        str: 比较报告文件路径
    """
    # 处理图像
    results = process_image_with_all_algorithms(input_path, output_folder)
    
    # 生成比较报告
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_folder, f"{input_filename}_comparison_report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"图像处理算法比较报告\n")
        f.write(f"输入图像: {input_path}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"算法比较:\n")
        f.write(f"1. 默认算法: 标准四色处理，不处理特殊颜色\n")
        f.write(f"2. 特殊颜色算法C: 使用特定抖动模式处理蓝色、绿色和青色\n")
        f.write(f"3. 特殊颜色算法T: 使用2x2抖动矩阵处理蓝色、绿色和青色\n\n")
        
        f.write(f"处理结果:\n")
        for image_path, hex_packets in results.items():
            algorithm_name = "未知"
            if "default" in image_path:
                algorithm_name = "默认算法"
            elif "special_C" in image_path:
                algorithm_name = "特殊颜色算法C"
            elif "special_T" in image_path:
                algorithm_name = "特殊颜色算法T"
                
            status = "成功" if hex_packets else "失败"
            packet_count = len(hex_packets) if hex_packets else 0
            
            f.write(f"- {algorithm_name}: {status}\n")
            f.write(f"  输出图像: {os.path.basename(image_path)}\n")
            f.write(f"  数据包数量: {packet_count}\n\n")
    
    print(f"比较报告已生成: {report_path}")
    return report_path

# 命令行参数处理
def main():
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='使用三种不同算法处理图像并转换为四色墨水屏格式')
    parser.add_argument('input_image', type=str, help='输入图像的路径')
    parser.add_argument('-o', '--output', type=str, default='output', help='输出文件夹路径，默认为当前目录下的output文件夹')
    parser.add_argument('-c', '--compare', action='store_true', help='是否生成算法比较报告')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 确保输出文件夹存在
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 处理图像
    if args.compare:
        # 处理图像并生成比较报告
        report_path = compare_algorithms(args.input_image, args.output)
        print(f"处理完成，比较报告已保存至: {report_path}")
    else:
        # 只处理图像，不生成比较报告
        results = process_image_with_all_algorithms(args.input_image, args.output)
        print("处理完成，输出图像:")
        for image_path in results.keys():
            print(f"- {image_path}")

# 示例用法
if __name__ == "__main__":
    main()