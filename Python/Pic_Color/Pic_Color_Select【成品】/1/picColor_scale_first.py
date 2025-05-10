# 通过使用 K-means 聚类算法对图像进行颜色聚类分析，并生成一个基于聚类中心(即最具代表性的颜色)的RGB值和调色板。
# 使用抖动算法对颜色进行处理
# 先缩放图像到200像素，再进行抖动处理

# 墨水屏像素定义：
# 黑 00
# 白 01
# 黄 10
# 红 11

import numpy as np
import cv2
from sklearn.cluster import KMeans
import os

def process_image_for_eink(input_path, output_image_path, output_packets_path=""):
    """
    处理图片为四色墨水屏格式并输出相应数据
    
    参数:
        input_path (str): 输入图片路径
        output_image_path (str): 输出处理后图片路径
        output_packets_path (str, optional): 输出二进制数据列表文件路径，默认为空字符串，为空时不生成数据包文件
        
    返回:
        list/bool: 成功时返回十六进制数据包列表，失败时返回False
    """
    try:
        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f'无法读取图片文件：{input_path}')
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取图像尺寸
        h, w, d = image.shape
        
        # 检查图片尺寸是否满足要求 (不小于200*200)
        if h < 200 or w < 200:
            raise ValueError(f'处理的图片尺寸({w}x{h})小于所需的200x200像素')
        
        # --- 1. 图像缩放 ---
        # 计算缩放比例，使较短的边等于200像素
        if w < h:  # 宽度更小
            scale_factor = 200 / w
            scaled_w = 200
            scaled_h = int(h * scale_factor)
        else:  # 高度更小或相等
            scale_factor = 200 / h
            scaled_h = 200
            scaled_w = int(w * scale_factor)
        
        # 缩放图像
        scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        
        # 更新尺寸变量
        h, w, d = scaled_image.shape
        
        # --- 2. K-means 颜色聚类分析 ---
        # 重塑图像为二维数组，每行代表一个像素的RGB值
        pixels = scaled_image.reshape(h * w, d)
        
        # 设置固定的颜色提取数量为10
        n_colors = 10
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels)
        
        # 获取聚类中心（调色板）
        palette = np.uint8(kmeans.cluster_centers_)
        
        # 定义电子开发板支持的颜色（RGB格式）
        target_colors = {
            '黑色': np.array([0, 0, 0]),
            '白色': np.array([255, 255, 255]),
            '红色': np.array([255, 0, 0]),
            '黄色': np.array([255, 255, 0])
        }
        
        # 将提取的颜色映射到目标颜色
        mapped_palette = []
        color_mapping = {}
        
        for i, color in enumerate(palette):
            # 计算与每个目标颜色的欧氏距离
            distances = {name: np.sqrt(np.sum((color - target_color) ** 2)) for name, target_color in target_colors.items()}
            
            # 找到距离最小的目标颜色
            closest_color_name = min(distances, key=distances.get)
            closest_color = target_colors[closest_color_name]
            
            mapped_palette.append(closest_color)
            color_mapping[i] = {
                '原始颜色': color,
                '映射颜色': closest_color,
                '颜色名称': closest_color_name,
                '距离': distances[closest_color_name]
            }
        
        # mapped_palette = np.array(mapped_palette)
        
        # --- 3. 抖动算法处理 ---
        
        # 将目标颜色转换为数组形式，便于计算
        target_colors_array = np.array(list(target_colors.values()))
        
        # 创建一个新的图像用于抖动处理
        dithered_image = scaled_image.copy().astype(np.float32)
        
        # 图像预处理 - 使用双边滤波进行边缘保持平滑
        dithered_image = cv2.bilateralFilter(dithered_image, d=5, sigmaColor=75, sigmaSpace=75)
        
        # 增强红色区域的识别 - 对红色分量明显高于其他分量的区域进行预处理
        for y in range(h):
            for x in range(w):
                pixel = dithered_image[y, x]
                # 如果红色分量明显高于绿色和蓝色分量，增强红色
                if pixel[0] > 170 and pixel[0] > pixel[1] * 1.5 and pixel[0] > pixel[2] * 1.5:
                    # 更强力地增强红色分量，降低其他分量
                    dithered_image[y, x, 0] = min(255, pixel[0] * 1.3)  # 更强力地增强红色
                    dithered_image[y, x, 1] = max(0, pixel[1] * 0.7)    # 更强力地降低绿色
                    dithered_image[y, x, 2] = max(0, pixel[2] * 0.7)    # 更强力地降低蓝色
                # 对于非常接近红色的区域，直接设置为纯红色
                elif pixel[0] > 220 and pixel[0] > pixel[1] * 2.0 and pixel[0] > pixel[2] * 2.0:
                    dithered_image[y, x] = np.array([255, 0, 0])  # 直接设置为纯红色
        
        # 定义Jarvis-Judice-Ninke抖动矩阵权重
        jjn_weights = [
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1]
        ]
        jjn_weights = np.array(jjn_weights) / 48  # 归一化权重
        
        # 定义颜色阈值，用于判断颜色是否足够接近目标颜色
        color_thresholds = {
            '黑色': 15,
            '白色': 15,
            '红色': 40,  # 红色区域使用更高的阈值，更容易被识别为纯色
            '黄色': 15
        }
        
        # 定义红色区域的特殊判断参数
        red_intensity_threshold = 180  # 红色分量强度阈值
        red_ratio_threshold = 1.7     # 红色与其他颜色分量的比例阈值
        
        # 应用改进的抖动算法
        total_pixels = h * w
        processed_pixels = 0
        
        for y in range(h):
            for x in range(w):
                # 获取当前像素的颜色（已经过预处理）
                old_pixel = dithered_image[y, x].copy()
                
                # 检查是否接近目标颜色，使用针对不同颜色的阈值
                is_close_to_target = False
                closest_target_color = None
                min_distance = float('inf')
                closest_color_name = None
                
                # 首先找到最接近的目标颜色
                for color_name, target_color in target_colors.items():
                    distance = np.sqrt(np.sum((old_pixel - target_color) ** 2))
                    if distance < min_distance:
                        min_distance = distance
                        closest_target_color = target_color
                        closest_color_name = color_name
                
                # 使用对应颜色的阈值判断是否足够接近
                if min_distance < color_thresholds[closest_color_name]:
                    is_close_to_target = True
                    dithered_image[y, x] = closest_target_color  # 直接设置为目标颜色
                    
                # 特殊处理红色区域：使用定义的参数判断是否为红色区域
                elif closest_color_name == '红色' and old_pixel[0] > red_intensity_threshold and old_pixel[0] > old_pixel[1] * red_ratio_threshold and old_pixel[0] > old_pixel[2] * red_ratio_threshold:
                    is_close_to_target = True
                    dithered_image[y, x] = target_colors['红色']  # 强制设为红色
                    
                # 额外检查：如果红色分量非常高，即使不是最接近的颜色也强制设为红色
                elif old_pixel[0] > 220 and old_pixel[0] > old_pixel[1] * 2.0 and old_pixel[0] > old_pixel[2] * 2.0:
                    is_close_to_target = True
                    dithered_image[y, x] = target_colors['红色']  # 强制设为红色
                
                # 如果颜色足够接近目标色，则跳过误差扩散
                if is_close_to_target:
                    processed_pixels += 1
                    continue  # 处理下一个像素
                
                # 如果颜色不够接近任何目标色，则执行标准抖动
                # 找到最接近的目标颜色
                if not is_close_to_target:
                    distances = np.sqrt(np.sum((old_pixel[np.newaxis, :] - target_colors_array) ** 2, axis=1))
                    closest_color_index = np.argmin(distances)
                    new_pixel = target_colors_array[closest_color_index]
                else:
                    # 理论上不会执行到这里，因为 continue 了
                    # 但为了代码完整性，保留查找逻辑
                    distances = np.sqrt(np.sum((old_pixel[np.newaxis, :] - target_colors_array) ** 2, axis=1))
                    closest_color_index = np.argmin(distances)
                    new_pixel = target_colors_array[closest_color_index]
                
                # 设置新像素颜色
                dithered_image[y, x] = new_pixel
                
                # 计算量化误差
                quant_error = old_pixel - new_pixel
                
                # 扩散误差到邻近像素
                for dy in range(3):
                    for dx in range(5):
                        weight = jjn_weights[dy, dx]
                        if weight == 0:
                            continue
                        
                        nx, ny = x + dx - 2, y + dy  # 计算邻近像素坐标 (dx-2是因为矩阵中心是(2,0))
                        
                        # 检查坐标是否在图像范围内
                        if 0 <= nx < w and 0 <= ny < h:
                            dithered_image[ny, nx] += quant_error * weight
                
                processed_pixels += 1
        
        # 将浮点数转换回uint8
        dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)
        
        # 确保所有像素都是目标颜色之一
        for y in range(h):
            for x in range(w):
                pixel = tuple(dithered_image[y, x])
                # 检查像素是否是目标颜色之一
                is_target_color = False
                for target_color in target_colors.values():
                    if np.array_equal(dithered_image[y, x], target_color):
                        is_target_color = True
                        break
                
                # 如果不是目标颜色，映射到最接近的目标颜色
                if not is_target_color:
                    min_distance = float('inf')
                    closest_color = None
                    for name, target_color in target_colors.items():
                        distance = np.sqrt(np.sum((dithered_image[y, x] - target_color) ** 2))
                        if distance < min_distance:
                            min_distance = distance
                            closest_color = target_color
                    
                    dithered_image[y, x] = closest_color
        
        # --- 4. 裁剪图像 ---
        # 获取抖动后图像的尺寸
        h_dith, w_dith, _ = dithered_image.shape
        
        # 自动以图片中心点为中心进行裁剪
        center_x, center_y = w_dith // 2, h_dith // 2
        start_x = max(0, center_x - 100)
        start_y = max(0, center_y - 100)
        
        # 调整以确保能裁剪到完整的200x200区域
        if start_x + 200 > w_dith:
            start_x = w_dith - 200
        if start_y + 200 > h_dith:
            start_y = h_dith - 200
        
        # 裁剪处理后的图像
        cropped_image = dithered_image[start_y:start_y+200, start_x:start_x+200]
        
        # 确保裁剪后的图像中所有像素都是目标颜色之一
        for y in range(200):
            for x in range(200):
                pixel = tuple(cropped_image[y, x])
                # 检查像素是否是目标颜色之一
                is_target_color = False
                for target_color in target_colors.values():
                    if np.array_equal(cropped_image[y, x], target_color):
                        is_target_color = True
                        break
                
                # 如果不是目标颜色，映射到最接近的目标颜色
                if not is_target_color:
                    min_distance = float('inf')
                    closest_color = None
                    for name, target_color in target_colors.items():
                        distance = np.sqrt(np.sum((cropped_image[y, x] - target_color) ** 2))
                        if distance < min_distance:
                            min_distance = distance
                            closest_color = target_color
                    
                    cropped_image[y, x] = closest_color
        
        # 保存裁剪后的图像
        cv2.imwrite(output_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        
        # --- 5. 将裁剪后的图像转换为墨水屏数据包 ---
        
        # 墨水屏像素定义：
        # 黑 00
        # 白 01
        # 黄 10
        # 红 11
        
        # 创建颜色到二进制值的映射
        color_to_binary = {
            '黑色': '00',
            '白色': '01',
            '黄色': '10',
            '红色': '11'
        }
        
        # 创建RGB值到颜色名称的映射
        rgb_to_color_name = {}
        for name, rgb in target_colors.items():
            rgb_tuple = tuple(rgb)
            rgb_to_color_name[rgb_tuple] = name
        
        # 创建一个数组来存储所有像素的二进制表示
        binary_pixels = []
        for y in range(200):
            for x in range(200):
                pixel = tuple(cropped_image[y, x])
                # 修复：如果像素不在映射表中，找到最接近的目标颜色
                if pixel not in rgb_to_color_name:
                    # 计算与每个目标颜色的欧氏距离
                    min_distance = float('inf')
                    closest_color = None
                    for name, target_color in target_colors.items():
                        target_tuple = tuple(target_color)
                        distance = np.sqrt(np.sum(np.array([(pixel[i] - target_tuple[i])**2 for i in range(3)])))
                        if distance < min_distance:
                            min_distance = distance
                            closest_color = name
                            # 直接修正像素值为目标颜色
                            cropped_image[y, x] = target_colors[closest_color]
                    color_name = closest_color
                else:
                    color_name = rgb_to_color_name[pixel]
                binary_pixels.append(color_to_binary[color_name])
        
        # 将二进制像素数据转换为字节
        byte_data = []
        current_byte = ''
        for binary_pixel in binary_pixels:
            current_byte += binary_pixel
            if len(current_byte) == 8:  # 当积累了8位（4个像素）时，转换为一个字节
                byte_value = int(current_byte, 2)
                byte_data.append(byte_value)
                current_byte = ''
        
        # 如果最后有不足8位的数据，补0并添加
        if current_byte:
            current_byte = current_byte.ljust(8, '0')
            byte_value = int(current_byte, 2)
            byte_data.append(byte_value)
        
        # 创建数据包
        packets = []
        
        # 创建前50个数据包 (0x00-0x31)
        for i in range(50):
            packet_number = i  # 包序号 (0x00-0x31)
            packet_data = byte_data[i * 200:(i + 1) * 200]  # 每包200字节图像数据
            
            # 计算校验和 (简单的字节求和)
            checksum = (packet_number + sum(packet_data)) % 256
            
            # 组装完整的数据包 (序号 + 数据 + 校验和)
            complete_packet = [packet_number] + packet_data + [checksum]
            packets.append(complete_packet)
        
        # 将数据包转为字符串列表格式
        hex_packets = []
        for packet in packets:
            packet_hex = ''.join([f"{byte:02X}" for byte in packet])
            hex_packets.append(packet_hex)
        
        # 如果提供了输出路径，则保存为txt文件
        if output_packets_path:
            # 保存为txt文件，格式为["123814080", "345738975", ...]
            with open(output_packets_path, 'w', encoding='utf-8') as f:
                f.write('[\n')
                for i, packet in enumerate(hex_packets):
                    if i < len(hex_packets) - 1:
                        f.write(f'    "{packet}",\n')
                    else:
                        f.write(f'    "{packet}"\n')
                f.write(']\n')
        
        return hex_packets
    
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return False


def process_hex_data(file_path):
    """
    处理十六进制数据文件，提取数据包信息
    
    参数:
        file_path (str): 十六进制数据文件路径
        
    返回:
        list: 提取的十六进制数据包列表
    """
    hex_packets = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检查是否是数据包标题行
        if line.startswith('数据包'):
            # 获取数据包编号
            packet_num = int(line.split('(')[1].split(')')[0].replace('0x', ''), 16)
            
            # 跳过最后一个固定的数据包 (0x32)
            if packet_num == 0x32:
                break
                
            # 下一行应该是数据行
            if i + 1 < len(lines):
                data_line = lines[i + 1].strip()
                # 移除所有空格
                hex_data = data_line.replace(' ', '')
                if hex_data:  # 确保不是空行
                    hex_packets.append(hex_data)
                i += 2  # 跳过数据行
            else:
                i += 1
        else:
            i += 1
    
    return hex_packets


# 示例用法
if __name__ == "__main__":
    # 示例：处理一张图片
    input_image = "/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/Pic_Color_Select【成品】/1/pic/2024-9.jpg"  # 输入图片路径
    output_image = "/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/Pic_Color_Select【成品】/1/pic/2024-9_output.jpg"  # 输出图片路径
    output_data = "data.txt"  # 输出数据包文件路径
        # 调用函数处理图片
    result = process_image_for_eink(input_image, output_image)
    #result = process_image_for_eink(input_image, output_image, output_data)
    if result:
        print(f"图片处理成功，生成了 {len(result)} 个数据包")
    else:
        print("图片处理失败")