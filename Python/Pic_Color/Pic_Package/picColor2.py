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
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os
import glob
import re

# 设置matplotlib中文字体 (如果需要在服务器环境运行且需要生成 matplotlib 图表，可能需要配置字体)
try:
    plt.rcParams['font.family'] = ['Hiragino Sans GB']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']
except Exception as e:
    print(f"无法设置中文字体，matplotlib 图表中的中文可能显示异常: {e}")

# 定义电子开发板支持的颜色（RGB格式）
TARGET_COLORS = {
    '黑色': np.array([0, 0, 0]),
    '白色': np.array([255, 255, 255]),
    '红色': np.array([255, 0, 0]),
    '黄色': np.array([255, 255, 0])
}

# 墨水屏像素定义到二进制值的映射
COLOR_TO_BINARY = {
    '黑色': '00',
    '白色': '01',
    '黄色': '10',
    '红色': '11'
}

# --- 核心处理函数 ---

def process_image(image_path, crop_mode, custom_coord=None):
    """
    处理图像，进行颜色聚类、抖动、缩放、裁剪，并生成墨水屏数据。

    Args:
        image_path (str): 输入图像的完整路径。
        crop_mode (int): 裁剪模式。0: 中心裁剪, 1: 自定义坐标裁剪。
        custom_coord (int, optional): 自定义裁剪坐标值。当 crop_mode 为 1 时必需。
                                      如果缩放后宽度 > 200，此值为 x 坐标 (从左侧开始)。
                                      如果缩放后高度 > 200，此值为 y 坐标 (从底部开始)。
                                      Defaults to None.

    Returns:
        dict: 包含处理结果的字典。
              {'final_image_path': str, 'hex_data_list': list[str]}
              如果处理失败则返回 None。
    Raises:
        FileNotFoundError: 如果图像文件不存在。
        ValueError: 如果图像尺寸过小或参数无效。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 读取和预处理图像 ---
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'图像文件不存在：{image_path}')

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'无法读取图片文件：{image_path}')

    # 转换为RGB格式
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取图像尺寸
    h, w, d = image_rgb.shape

    # 检查图片尺寸是否满足要求 (不小于200*200)
    if h < 200 or w < 200:
        raise ValueError(f'处理的图片尺寸({w}x{h})小于所需的200x200像素')

    # --- 2. K-means 颜色聚类 (可选，但抖动需要映射到目标色) ---
    # 重塑图像为二维数组
    pixels = image_rgb.reshape(h * w, d)
    n_colors = 10 # 可以调整或移除，因为最终只映射到4种目标色
    # kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels)
    # palette = np.uint8(kmeans.cluster_centers_)
    # K-means 部分可以省略，因为抖动算法会直接将像素映射到最近的目标颜色

    # --- 3. 抖动算法处理 ---
    target_colors_array = np.array(list(TARGET_COLORS.values()))
    dithered_image_float = image_rgb.copy().astype(np.float32)

    # 图像预处理 - 双边滤波
    dithered_image_float = cv2.bilateralFilter(dithered_image_float, d=5, sigmaColor=75, sigmaSpace=75)

    # 定义Jarvis-Judice-Ninke抖动矩阵权重
    jjn_weights = [
        [0, 0, 0, 7, 5],
        [3, 5, 7, 5, 3],
        [1, 3, 5, 3, 1]
    ]
    jjn_weights = np.array(jjn_weights) / 48.0 # 使用浮点数除法

    # 应用抖动算法
    for y in range(h):
        for x in range(w):
            old_pixel = dithered_image_float[y, x].copy()

            # 考虑局部区域颜色 (可选优化)
            # local_region = dithered_image_float[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
            # local_mean = np.mean(local_region, axis=(0, 1))
            # old_pixel = old_pixel * 0.7 + local_mean * 0.3

            # 找到最接近的目标颜色
            distances = np.sqrt(np.sum((old_pixel - target_colors_array) ** 2, axis=1))
            closest_idx = np.argmin(distances)
            new_pixel = target_colors_array[closest_idx]

            # 更新当前像素
            dithered_image_float[y, x] = new_pixel

            # 计算量化误差
            quant_error = old_pixel - new_pixel

            # 分散误差
            for dy in range(3):
                for dx_offset in range(5):
                    weight = jjn_weights[dy, dx_offset]
                    if weight > 0:
                        nx = x + dx_offset - 2 # -2 使矩阵中心对齐
                        ny = y + dy
                        if 0 <= ny < h and 0 <= nx < w:
                            dithered_image_float[ny, nx] += quant_error * weight

    # 将浮点数转换回uint8
    dithered_image = np.clip(dithered_image_float, 0, 255).astype(np.uint8)

    # --- 4. 缩放图像 ---
    h, w, _ = dithered_image.shape # 获取抖动后图像的尺寸
    scaled_w, scaled_h = w, h
    scale_factor = 1.0
    scaled_axis = None # 记录哪个轴被缩放到了200

    if w < h: # 宽度更小
        if w != 200:
            scale_factor = 200 / w
            scaled_w = 200
            scaled_h = int(h * scale_factor)
            scaled_axis = 'width'
    else: # 高度更小或相等
        if h != 200:
            scale_factor = 200 / h
            scaled_h = 200
            scaled_w = int(w * scale_factor)
            scaled_axis = 'height'

    # 只有在尺寸需要改变时才缩放
    if scaled_w != w or scaled_h != h:
        scaled_image = cv2.resize(dithered_image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    else:
        scaled_image = dithered_image # 无需缩放

    # 更新尺寸
    h_scaled, w_scaled = scaled_image.shape[:2]

    # --- 5. 裁剪图像 ---
    start_x = 0
    start_y = 0

    if w_scaled == 200 and h_scaled == 200:
        # 无需裁剪
        cropped_image = scaled_image
    elif crop_mode == 0: # 中心裁剪
        center_x, center_y = w_scaled // 2, h_scaled // 2
        start_x = max(0, center_x - 100)
        start_y = max(0, center_y - 100)
        # 确保能裁剪到200x200
        if start_x + 200 > w_scaled:
            start_x = w_scaled - 200
        if start_y + 200 > h_scaled:
            start_y = h_scaled - 200
        cropped_image = scaled_image[start_y:start_y + 200, start_x:start_x + 200]

    elif crop_mode == 1: # 自定义坐标裁剪
        if custom_coord is None:
            raise ValueError("自定义裁剪模式需要提供 custom_coord 参数")
        custom_coord = int(custom_coord)

        if w_scaled > 200 and h_scaled == 200: # 宽度 > 200, 高度 = 200
            max_x = w_scaled - 200
            if not (0 <= custom_coord <= max_x):
                raise ValueError(f"自定义 x 坐标 {custom_coord} 超出范围 [0, {max_x}]")
            start_x = custom_coord
            start_y = 0
        elif h_scaled > 200 and w_scaled == 200: # 高度 > 200, 宽度 = 200
            max_y_bottom = h_scaled - 200
            if not (0 <= custom_coord <= max_y_bottom):
                raise ValueError(f"自定义 y 坐标 {custom_coord} (从底部算起) 超出范围 [0, {max_y_bottom}]")
            # 将从底部开始的坐标转换为从顶部开始的坐标
            start_y = h_scaled - 200 - custom_coord
            start_x = 0
        elif w_scaled > 200 and h_scaled > 200: # 宽度和高度都 > 200
             # 根据哪个轴被缩放来决定 custom_coord 应用于哪个轴
            if scaled_axis == 'width': # 宽度被缩放到200，高度 > 200，custom_coord 应用于 Y
                max_y_bottom = h_scaled - 200
                if not (0 <= custom_coord <= max_y_bottom):
                    raise ValueError(f"自定义 y 坐标 {custom_coord} (从底部算起) 超出范围 [0, {max_y_bottom}]")
                start_y = h_scaled - 200 - custom_coord
                start_x = 0 # 宽度已经是200，x从0开始
            elif scaled_axis == 'height': # 高度被缩放到200，宽度 > 200，custom_coord 应用于 X
                max_x = w_scaled - 200
                if not (0 <= custom_coord <= max_x):
                    raise ValueError(f"自定义 x 坐标 {custom_coord} 超出范围 [0, {max_x}]")
                start_x = custom_coord
                start_y = 0 # 高度已经是200，y从0开始
            else: # 理论上不会发生，因为如果都大于200，必有一个被缩放
                 raise RuntimeError("无法确定自定义坐标应用于哪个轴")
        else: # w_scaled <= 200 and h_scaled <= 200 (至少一个等于200，另一个小于等于200)
            # 这种情况不应该发生，因为原始尺寸检查会过滤掉小于200x200的图片
            # 并且缩放逻辑会保证至少一个维度是200
            raise RuntimeError("缩放或裁剪逻辑出现意外情况")

        cropped_image = scaled_image[start_y:start_y + 200, start_x:start_x + 200]
    else:
        raise ValueError(f"无效的裁剪模式: {crop_mode}")

    # --- 6. 保存最终图像 ---
    final_image_path = os.path.join(output_dir, 'final_image_200x200.png')
    # 保存前确保图像是 BGR 格式
    if cropped_image.shape[2] == 3: # 彩色图
        cv2.imwrite(final_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
    else: # 灰度图 (理论上不会是灰度图)
        cv2.imwrite(final_image_path, cropped_image)

    # --- 7. 生成墨水屏数据包 ---
    # 辅助函数：找到最接近的目标颜色名称
    def find_closest_color_name(pixel):
        pixel_array = np.array(pixel)
        distances = {name: np.sqrt(np.sum((pixel_array - color) ** 2)) for name, color in TARGET_COLORS.items()}
        return min(distances, key=distances.get)

    # 将像素转换为二进制字符串
    binary_pixels_flat = []
    for y in range(200):
        for x in range(200):
            pixel = tuple(cropped_image[y, x])
            color_name = find_closest_color_name(pixel)
            binary_pixels_flat.append(COLOR_TO_BINARY[color_name])

    # 将二进制字符串转换为字节列表
    byte_data = []
    current_byte_str = ''
    for binary_pixel in binary_pixels_flat:
        current_byte_str += binary_pixel
        if len(current_byte_str) == 8:
            byte_data.append(int(current_byte_str, 2))
            current_byte_str = ''
    # 处理末尾不足8位的情况 (理论上 200*200*2 正好是8的倍数，不会发生)
    # if current_byte_str:
    #     current_byte_str = current_byte_str.ljust(8, '0')
    #     byte_data.append(int(current_byte_str, 2))

    # 创建数据包 (共51个包)
    packets_raw = []
    # 前50个图像数据包
    for i in range(50):
        packet_number = i
        packet_payload = byte_data[i * 200:(i + 1) * 200]
        checksum = (packet_number + sum(packet_payload)) % 256
        packets_raw.append([packet_number] + packet_payload + [checksum])

    # 最后一个命令包
    last_packet_number = 0x32
    last_packet_payload = [ord(c) for c in "screen updating\r\n"]
    last_packet_checksum = (last_packet_number + sum(last_packet_payload)) % 256
    # 命令包也需要填充到201字节 (1序号 + payload + 1校验和 + 填充)
    padding_len = 201 - 1 - len(last_packet_payload) - 1 # 实际payload是200字节，所以填充200-len(payload)
    last_packet_payload_padded = last_packet_payload + [0] * (200 - len(last_packet_payload)) # 填充到200字节
    packets_raw.append([last_packet_number] + last_packet_payload_padded + [last_packet_checksum])


    # 格式化为十六进制字符串列表
    hex_data_list = []
    for packet in packets_raw:
        # 移除首尾的序号和校验和，只取中间的 payload (packet[1]到packet[-2])
        payload_bytes = packet[1:-1]
        # 将 payload bytes 转换为无空格的十六进制字符串
        hex_string = ''.join([f"{byte:02X}" for byte in payload_bytes])
        # 在前面加上包序号的十六进制表示
        packet_hex_prefix = f"{packet[0]:02X}"
        hex_data_list.append(packet_hex_prefix + hex_string)

    # --- 8. 返回结果 ---
    return {
        'final_image_path': final_image_path,
        'hex_data_list': hex_data_list
    }

# --- 示例用法 ---
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
                hex_output_path = os.path.join(output_dir, 'eink_display_data_hex_new.txt')
                with open(hex_output_path, 'w') as f_hex:
                    for i, hex_line in enumerate(result_center['hex_data_list']):
                         packet_index_hex = hex_line[:2]
                         payload_hex = hex_line[2:]
                         # 重新格式化为带空格的，以便对比
                         payload_spaced = ' '.join(payload_hex[j:j+2] for j in range(0, len(payload_hex), 2))
                         f_hex.write(f"数据包 {i} (0x{packet_index_hex}):\n")
                         f_hex.write(payload_spaced + '\n\n')
                print(f"新的 Hex 数据已写入: {hex_output_path}")


        except Exception as e:
            print(f"中心裁剪处理失败: {e}")

        # 示例2: 自定义裁剪 (假设缩放后高度>200, 裁剪底部向上10个像素的位置)
        # 注意：需要根据实际图片缩放情况调整 custom_coord 和说明
        try:
            print("\n--- 示例 2: 自定义裁剪 (假设 y=10 从底部算起) ---")
            # 需要先知道图片缩放后的尺寸才能确定 custom_coord 的有效范围和含义
            # 这里假设缩放后高度 > 200, 宽度 = 200
            # custom_coord=10 表示从底部向上数10个像素的位置开始裁剪
            result_custom = process_image(example_image_path, crop_mode=1, custom_coord=10)
            if result_custom:
                print(f"最终图片保存在: {result_custom['final_image_path']}")
                print(f"生成了 {len(result_custom['hex_data_list'])} 个数据包")
        except ValueError as e:
             print(f"自定义裁剪处理失败: {e} (可能是坐标无效，请根据图片调整)")
        except Exception as e:
            print(f"自定义裁剪处理失败: {e}")

    else:
        print(f"在 {photos_dir} 中未找到示例图片 (photo*.jpg/png/bmp)")