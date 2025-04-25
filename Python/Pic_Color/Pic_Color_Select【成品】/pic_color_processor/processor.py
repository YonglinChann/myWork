# -*- coding: utf-8 -*-
# Core image processing logic

import numpy as np
import cv2
import os

# Define target colors and binary mapping (moved from picColor2.py)
TARGET_COLORS = {
    '黑色': np.array([0, 0, 0]),
    '白色': np.array([255, 255, 255]),
    '红色': np.array([255, 0, 0]),
    '黄色': np.array([255, 255, 0])
}

COLOR_TO_BINARY = {
    '黑色': '00',
    '白色': '01',
    '黄色': '10',
    '红色': '11'
}

def process_image(image_path, crop_mode, custom_coord=None, output_dir_base=None):
    """
    处理图像，进行颜色聚类、抖动、缩放、裁剪，并生成墨水屏数据。

    Args:
        image_path (str): 输入图像的完整路径。
        crop_mode (int): 裁剪模式。0: 中心裁剪, 1: 自定义坐标裁剪。
        custom_coord (int, optional): 自定义裁剪坐标值。当 crop_mode 为 1 时必需。
                                      如果缩放后宽度 > 200，此值为 x 坐标 (从左侧开始)。
                                      如果缩放后高度 > 200，此值为 y 坐标 (从底部开始)。
                                      Defaults to None.
        output_dir_base (str, optional): 指定输出目录的基础路径。如果为 None，则默认为库文件所在目录下的 'output'。
                                         Defaults to None.

    Returns:
        dict: 包含处理结果的字典。
              {'final_image_path': str, 'hex_data_list': list[str]}
              如果处理失败则返回 None。
    Raises:
        FileNotFoundError: 如果图像文件不存在。
        ValueError: 如果图像尺寸过小或参数无效。
    """
    # Determine output directory
    if output_dir_base is None:
        # Default to 'output' directory relative to this processor file if not specified
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output') # Go up one level from pic_color_processor
    else:
        output_dir = os.path.join(output_dir_base, 'output')
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

    # --- 2. K-means 颜色聚类 (Optional, as dithering maps directly) ---
    # K-means part can be omitted as dithering maps directly to target colors

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
    h_dith, w_dith, _ = dithered_image.shape # Use dithered image dimensions
    scaled_w, scaled_h = w_dith, h_dith
    scale_factor = 1.0
    scaled_axis = None # 记录哪个轴被缩放到了200

    if w_dith < h_dith: # 宽度更小
        if w_dith != 200:
            scale_factor = 200 / w_dith
            scaled_w = 200
            scaled_h = int(h_dith * scale_factor)
            scaled_axis = 'width'
    else: # 高度更小或相等
        if h_dith != 200:
            scale_factor = 200 / h_dith
            scaled_h = 200
            scaled_w = int(w_dith * scale_factor)
            scaled_axis = 'height'

    # 只有在尺寸需要改变时才缩放
    if scaled_w != w_dith or scaled_h != h_dith:
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
        try:
            custom_coord = int(custom_coord)
        except ValueError:
             raise ValueError(f"无效的自定义坐标值: {custom_coord}")

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
        else: # w_scaled <= 200 and h_scaled <= 200 (at least one is 200)
            # This case should ideally not happen due to initial size check and scaling logic
            # If it does, treat as no crop needed if already 200x200, else raise error
            if w_scaled == 200 and h_scaled == 200:
                 cropped_image = scaled_image # Already correct size
            else:
                raise RuntimeError(f"缩放或裁剪逻辑出现意外情况: scaled size {w_scaled}x{h_scaled}")

        # Perform the crop only if needed (i.e., not already 200x200)
        if not (w_scaled == 200 and h_scaled == 200):
             cropped_image = scaled_image[start_y:start_y + 200, start_x:start_x + 200]

    else:
        raise ValueError(f"无效的裁剪模式: {crop_mode}")

    # Ensure cropped_image is defined
    if 'cropped_image' not in locals():
         raise RuntimeError("Cropped image was not generated correctly.")

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
            # Ensure pixel access is within bounds (should be 200x200)
            if y < cropped_image.shape[0] and x < cropped_image.shape[1]:
                pixel = tuple(cropped_image[y, x])
                color_name = find_closest_color_name(pixel)
                binary_pixels_flat.append(COLOR_TO_BINARY[color_name])
            else:
                 # Handle potential edge cases if cropping logic had issues, though it shouldn't
                 print(f"Warning: Accessing pixel out of bounds at ({x},{y}) in a {cropped_image.shape[0]}x{cropped_image.shape[1]} image. Using black.")
                 binary_pixels_flat.append(COLOR_TO_BINARY['黑色'])


    # 将二进制字符串转换为字节列表
    byte_data = []
    current_byte_str = ''
    for binary_pixel in binary_pixels_flat:
        current_byte_str += binary_pixel
        if len(current_byte_str) == 8:
            byte_data.append(int(current_byte_str, 2))
            current_byte_str = ''
    # 200*200*2 bits = 80000 bits = 10000 bytes. No leftover bits expected.

    # 创建数据包 (共51个包)
    packets_raw = []
    bytes_per_packet = 200
    num_image_packets = 50 # 10000 bytes / 200 bytes/packet

    # 前50个图像数据包
    for i in range(num_image_packets):
        packet_number = i
        packet_payload = byte_data[i * bytes_per_packet:(i + 1) * bytes_per_packet]
        if len(packet_payload) != bytes_per_packet:
             raise RuntimeError(f"图像数据包 {i} 长度错误: 预期 {bytes_per_packet}, 得到 {len(packet_payload)}")
        checksum = (packet_number + sum(packet_payload)) % 256
        packets_raw.append([packet_number] + packet_payload + [checksum])

    # 最后一个命令包
    last_packet_number = 0x32 # 50 in decimal
    last_packet_payload_str = "screen updating\r\n"
    last_packet_payload = [ord(c) for c in last_packet_payload_str]
    # 命令包也需要填充到 payload 长度为 200 字节
    padding_len = bytes_per_packet - len(last_packet_payload)
    last_packet_payload_padded = last_packet_payload + [0] * padding_len
    last_packet_checksum = (last_packet_number + sum(last_packet_payload_padded)) % 256 # Checksum uses padded payload
    packets_raw.append([last_packet_number] + last_packet_payload_padded + [last_packet_checksum])

    # 格式化为十六进制字符串列表 (包序号 + payload)
    hex_data_list = []
    for packet in packets_raw:
        packet_index_byte = packet[0]
        payload_bytes = packet[1:-1] # Exclude index and checksum
        # 将 payload bytes 转换为无空格的十六进制字符串
        hex_string = ''.join([f"{byte:02X}" for byte in payload_bytes])
        # 在前面加上包序号的十六进制表示 (2位)
        packet_hex_prefix = f"{packet_index_byte:02X}"
        hex_data_list.append(packet_hex_prefix + hex_string)

    # --- 8. 返回结果 ---
    return {
        'final_image_path': final_image_path,
        'hex_data_list': hex_data_list
    }