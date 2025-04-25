# -*- coding: utf-8 -*-
# 通过使用 K-means 聚类算法对图像进行颜色聚类分析，并生成一个基于聚类中心(即最具代表性的颜色)的RGB值和调色板。
# 使用抖动算法对颜色进行处理
# 导入必要的库

# 墨水屏像素定义：
# 黑 00
# 白 01
# 黄 10
# 红 11

import numpy as np
import matplotlib.pyplot as plt # Keep for potential future plotting or remove if unused
import cv2 # Keep for potential future image loading/display or remove if unused
import os
import glob
# import re # No longer needed here
# from sklearn.cluster import KMeans # No longer needed here

# Import the processing function from the new library
from pic_color_processor import process_image

# 设置matplotlib中文字体 (如果需要在服务器环境运行且需要生成 matplotlib 图表，可能需要配置字体)
try:
    plt.rcParams['font.family'] = ['Hiragino Sans GB']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
except Exception as e:
    print(f"无法设置中文字体，matplotlib 图表中的中文可能显示异常: {e}")

# --- 示例用法 --- (Now uses the imported library function)
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    photos_dir = os.path.join(current_dir, 'photos')

    # 查找示例图片
    example_image_path = None
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        found = glob.glob(os.path.join(photos_dir, f"photo*[0-9].{ext}"))
        if found:
            # 尝试找 photo3.png，否则用找到的第一个
            specific_path = os.path.join(photos_dir, 'photo3.png')
            if os.path.exists(specific_path):
                example_image_path = specific_path
                break
            else:
                example_image_path = sorted(found)[0] # 按字母顺序取第一个
                break

    if example_image_path:
        print(f"使用示例图片: {example_image_path}")

        # 示例1: 中心裁剪
        try:
            print("\n--- 示例 1: 中心裁剪 ---")
            result_center = process_image(example_image_path, crop_mode=0)
            if result_center:
                print(f"最终图片保存在: {result_center['final_image_path']}")
                print(f"生成了 {len(result_center['hex_data_list'])} 个数据包")
                print("部分十六进制数据:")
                for i in range(min(3, len(result_center['hex_data_list']))):
                    print(f"  包 {i}: {result_center['hex_data_list'][i][:60]}...") # 显示前60个字符
                # print(f"  包 46: {result_center['hex_data_list'][46]}") # 打印指定包的完整数据
                # print(f"  包 47: {result_center['hex_data_list'][47]}")
                print(f"  最后一个包 (命令): {result_center['hex_data_list'][-1]}")

                # 写入 hex 数据到文件，与之前的格式对比
                # Note: The output directory is now determined inside process_image
                # We might need the final_image_path to determine the output dir if needed outside
                output_dir_used = os.path.dirname(result_center['final_image_path'])
                hex_output_path = os.path.join(output_dir_used, 'eink_display_data_hex_from_lib.txt')
                with open(hex_output_path, 'w') as f_hex:
                    for i, hex_line in enumerate(result_center['hex_data_list']):
                         packet_index_hex = hex_line[:2]
                         payload_hex = hex_line[2:]
                         # 重新格式化为带空格的，以便对比
                         payload_spaced = ' '.join(payload_hex[j:j+2] for j in range(0, len(payload_hex), 2))
                         f_hex.write(f"数据包 {i} (0x{packet_index_hex}):\n")
                         f_hex.write(payload_spaced + '\n\n')
                print(f"新的 Hex 数据已写入: {hex_output_path}")

        except FileNotFoundError as e:
            print(f"处理失败: {e}")
        except ValueError as e:
            print(f"处理失败: {e}")
        except Exception as e:
            print(f"中心裁剪处理失败: {e}")

        # 示例2: 自定义裁剪 (假设缩放后高度>200, 裁剪底部向上10个像素的位置)
        # 注意：需要根据实际图片缩放情况调整 custom_coord 和说明
        try:
            print("\n--- 示例 2: 自定义裁剪 (假设 y=10 从底部算起) ---")
            # 需要先知道图片缩放后的尺寸才能确定 custom_coord 的有效范围和含义
            # 这里假设缩放后高度 > 200, 宽度 = 200
            # custom_coord=10 表示从底部向上数10个像素的位置开始裁剪
            # Pass output_dir_base explicitly if you want control, otherwise it defaults
            result_custom = process_image(example_image_path, crop_mode=1, custom_coord=10, output_dir_base=current_dir)
            if result_custom:
                print(f"最终图片保存在: {result_custom['final_image_path']}")
                print(f"生成了 {len(result_custom['hex_data_list'])} 个数据包")
        except FileNotFoundError as e:
            print(f"处理失败: {e}")
        except ValueError as e:
             print(f"自定义裁剪处理失败: {e} (可能是坐标无效，请根据图片调整)")
        except Exception as e:
            print(f"自定义裁剪处理失败: {e}")

    else:
        print(f"在 {photos_dir} 中未找到示例图片 (photo*.jpg/png/bmp)")