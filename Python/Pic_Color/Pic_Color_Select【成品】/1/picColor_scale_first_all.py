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

def process_image_for_eink_default(input_path, output_image_path, output_packets_path=""):
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


def process_hex_data_default(file_path):
    """
    处理十六进制数据文件，提取数据包信息
    
    参数:
        file_path (str): 十六进制数据文件路径
        
    返回:
        list: 提取的十六进制数据包列表
    """
    hex_packets_default = []
    
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
                    hex_packets_default.append(hex_data)
                i += 2  # 跳过数据行
            else:
                i += 1
        else:
            i += 1
    
    print(f"{hex_packets_default}")
    return hex_packets_default




def identify_special_color(pixel):
    """
    识别像素是否属于特殊颜色（蓝色、绿色、青色）
    
    参数:
        pixel: 像素RGB值
        
    返回:
        str/None: 特殊颜色类型名称，如果不是特殊颜色则返回None
    """
    r, g, b = pixel
    
    # 检测蓝色
    if b > 150 and b > r * 1.5 and b > g * 1.5:
        if b < 200:  # 深蓝色
            return '深蓝'
        else:  # 浅蓝色
            return '浅蓝'
    
    # 检测绿色
    if g > 150 and g > r * 1.5 and g > b * 1.5:
        if g < 200:  # 深绿色
            return '深绿'
        else:  # 浅绿色
            return '浅绿'
    
    # 检测青色(蓝绿色)
    if g > 120 and b > 120 and abs(g - b) < 40 and g > r * 1.5 and b > r * 1.5:
        return '青色'
    
    return None

def process_image_for_eink_T(input_path, output_image_path, output_packets_path=""):
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
        # 同时添加蓝色、绿色和青色的特殊处理
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
                
                # 蓝色增强处理 - 对蓝色分量明显高于其他分量的区域进行预处理
                elif pixel[2] > 150 and pixel[2] > pixel[0] * 1.5 and pixel[2] > pixel[1] * 1.5:
                    # 标记为蓝色区域，后续会使用特殊的抖动模式
                    # 增强蓝色对比度，使其更容易被识别
                    dithered_image[y, x, 0] = max(0, pixel[0] * 0.5)     # 降低红色
                    dithered_image[y, x, 1] = max(0, pixel[1] * 0.5)     # 降低绿色
                    dithered_image[y, x, 2] = min(255, pixel[2] * 1.2)   # 增强蓝色
                
                # 绿色增强处理 - 对绿色分量明显高于其他分量的区域进行预处理
                elif pixel[1] > 150 and pixel[1] > pixel[0] * 1.5 and pixel[1] > pixel[2] * 1.5:
                    # 标记为绿色区域，后续会使用特殊的抖动模式
                    # 增强绿色对比度，使其更容易被识别
                    dithered_image[y, x, 0] = max(0, pixel[0] * 0.5)     # 降低红色
                    dithered_image[y, x, 1] = min(255, pixel[1] * 1.2)   # 增强绿色
                    dithered_image[y, x, 2] = max(0, pixel[2] * 0.5)     # 降低蓝色
                
                # 青色(蓝绿色)增强处理 - 蓝色和绿色分量都高且接近，红色分量低
                elif pixel[1] > 120 and pixel[2] > 120 and abs(pixel[1] - pixel[2]) < 40 and pixel[1] > pixel[0] * 1.5 and pixel[2] > pixel[0] * 1.5:
                    # 标记为青色区域，后续会使用特殊的抖动模式
                    # 增强青色对比度
                    dithered_image[y, x, 0] = max(0, pixel[0] * 0.3)     # 大幅降低红色
                    dithered_image[y, x, 1] = min(255, pixel[1] * 1.1)   # 轻微增强绿色
                    dithered_image[y, x, 2] = min(255, pixel[2] * 1.1)   # 轻微增强蓝色
        
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
        
        # 定义蓝色、绿色和青色的特殊判断参数
        blue_intensity_threshold = 150  # 蓝色分量强度阈值
        blue_ratio_threshold = 1.5     # 蓝色与其他颜色分量的比例阈值
        
        green_intensity_threshold = 150  # 绿色分量强度阈值
        green_ratio_threshold = 1.5     # 绿色与其他颜色分量的比例阈值
        
        cyan_intensity_threshold = 120   # 青色分量强度阈值
        cyan_diff_threshold = 40        # 蓝色和绿色分量差异阈值
        cyan_ratio_threshold = 1.5      # 青色与红色分量的比例阈值
        
        # 定义特殊颜色的抖动模式 - 使用2x2的抖动矩阵
        # 这些模式将用于在抖动过程中模拟蓝色、绿色和青色
        dither_patterns = {
            # 深蓝色：使用黑色和黄色的组合
            '深蓝': np.array([
                [target_colors['黑色'], target_colors['黄色']],
                [target_colors['黄色'], target_colors['黑色']]
            ]),
            
            # 浅蓝色：使用白色和黄色的组合
            '浅蓝': np.array([
                [target_colors['白色'], target_colors['黄色']],
                [target_colors['黄色'], target_colors['白色']]
            ]),
            
            # 深绿色：使用黑色和黄色的不同比例组合
            '深绿': np.array([
                [target_colors['黄色'], target_colors['黄色']],
                [target_colors['黑色'], target_colors['黄色']]
            ]),
            
            # 浅绿色：使用白色和黄色的不同比例组合
            '浅绿': np.array([
                [target_colors['黄色'], target_colors['黄色']],
                [target_colors['白色'], target_colors['黄色']]
            ]),
            
            # 青色：使用黄色和白色的特殊组合
            '青色': np.array([
                [target_colors['白色'], target_colors['黄色']],
                [target_colors['黄色'], target_colors['白色']]
            ])
        }
        
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
                special_color_type = None  # 用于标记特殊颜色类型
                
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
                
                # 使用identify_special_color函数检测特殊颜色
                elif special_color_type is None:  # 如果之前没有识别为特殊颜色
                    special_color_type = identify_special_color(old_pixel)
                
                # 如果颜色足够接近目标色，则跳过误差扩散
                if is_close_to_target:
                    processed_pixels += 1
                    continue  # 处理下一个像素
                
                # 如果是特殊颜色类型（蓝色、绿色或青色），应用特殊的抖动模式
                if special_color_type is not None:
                    # 获取对应的抖动模式
                    pattern = dither_patterns[special_color_type]
                    
                    # 根据像素坐标在2x2模式中选择对应的颜色
                    pattern_x = x % 2
                    pattern_y = y % 2
                    new_pixel = pattern[pattern_y, pattern_x]
                    
                    # 设置新像素颜色
                    dithered_image[y, x] = new_pixel
                    
                    # 不进行误差扩散，直接处理下一个像素
                    processed_pixels += 1
                    continue
                
                # 如果颜色不够接近任何目标色且不是特殊颜色，则执行标准抖动
                # 找到最接近的目标颜色
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
        
        # 确保所有像素都是目标颜色之一，同时保持特殊颜色区域的一致性
        for y in range(h):
            for x in range(w):
                pixel = tuple(dithered_image[y, x])
                # 检查像素是否是目标颜色之一
                is_target_color = False
                for target_color in target_colors.values():
                    if np.array_equal(dithered_image[y, x], target_color):
                        is_target_color = True
                        break
                
                # 如果不是目标颜色，检查是否是特殊颜色区域
                if not is_target_color:
                    # 检查周围像素，判断是否属于特殊颜色区域
                    special_color_type = identify_special_color(dithered_image[y, x])
                    
                    if special_color_type is not None:
                        # 如果是特殊颜色，使用对应的抖动模式
                        pattern = dither_patterns[special_color_type]
                        pattern_x = x % 2
                        pattern_y = y % 2
                        dithered_image[y, x] = pattern[pattern_y, pattern_x]
                    else:
                        # 如果不是特殊颜色，映射到最接近的目标颜色
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
        
        # 确保裁剪后的图像中所有像素都是目标颜色之一，同时保持特殊颜色区域的一致性
        for y in range(200):
            for x in range(200):
                pixel = tuple(cropped_image[y, x])
                # 检查像素是否是目标颜色之一
                is_target_color = False
                for target_color in target_colors.values():
                    if np.array_equal(cropped_image[y, x], target_color):
                        is_target_color = True
                        break
                
                # 如果不是目标颜色，检查是否是特殊颜色区域
                if not is_target_color:
                    # 检查是否属于特殊颜色区域
                    special_color_type = identify_special_color(cropped_image[y, x])
                    
                    if special_color_type is not None:
                        # 如果是特殊颜色，使用对应的抖动模式
                        pattern = dither_patterns[special_color_type]
                        pattern_x = x % 2
                        pattern_y = y % 2
                        cropped_image[y, x] = pattern[pattern_y, pattern_x]
                    else:
                        # 如果不是特殊颜色，映射到最接近的目标颜色
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


def process_hex_data_T(file_path):
    """
    处理十六进制数据文件，提取数据包信息
    
    参数:
        file_path (str): 十六进制数据文件路径
        
    返回:
        list: 提取的十六进制数据包列表
    """
    hex_packets_T = []
    
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
                    hex_packets_T.append(hex_data)
                i += 2  # 跳过数据行
            else:
                i += 1
        else:
            i += 1
    
    return hex_packets_T


def process_image_for_eink_C(input_path, output_image_path, output_packets_path=""):
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
        # 定义一个标准化颜色的函数，确保精确匹配目标颜色
        def standardize_color(pixel, target_colors):
            """
            确保颜色是精确的目标颜色之一
            
            参数:
                pixel: 要检查的像素颜色
                target_colors: 目标颜色字典
                
            返回:
                标准化后的颜色数组
            """
            # 转换为元组以便比较
            if isinstance(pixel, np.ndarray):
                pixel_tuple = tuple(pixel)
            else:
                pixel_tuple = pixel
                
            # 检查是否已经是精确的目标颜色
            for color_name, color_value in target_colors.items():
                color_tuple = tuple(color_value)
                if pixel_tuple == color_tuple:
                    return color_value
            
            # 如果不是精确的目标颜色，找到最接近的
            min_distance = float('inf')
            closest_color = None
            for color_name, color_value in target_colors.items():
                color_tuple = tuple(color_value)
                # 使用平方和距离而不是欧氏距离，避免浮点数精度问题
                distance = sum((pixel_tuple[i] - color_tuple[i])**2 for i in range(3))
                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_value
            
            return closest_color

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
        
        # 定义电子开发板支持的颜色（RGB格式）- 使用精确的整数值
        target_colors = {
            '黑色': np.array([0, 0, 0], dtype=np.uint8),
            '白色': np.array([255, 255, 255], dtype=np.uint8),
            '红色': np.array([255, 0, 0], dtype=np.uint8),
            '黄色': np.array([255, 255, 0], dtype=np.uint8)
        }
        
        # ==== 方案一: 添加蓝色和绿色的特殊处理函数 ====
        def map_blue_green_colors(pixel, target_colors):
            # 提取RGB分量
            r, g, b = pixel
            
            # 检测蓝色 - 蓝色分量明显高于其他分量
            if b > 150 and b > r * 1.5 and b > g * 1.5:
                # 对于深蓝色，使用黑色和黄色的抖动组合
                if b < 200:
                    # 返回一个标记，表示这是需要特殊抖动处理的蓝色
                    return "蓝色_深", None
                else:
                    # 浅蓝色使用黄色和白色的抖动组合
                    return "蓝色_浅", None
            
            # 检测绿色 - 绿色分量明显高于其他分量
            if g > 150 and g > r * 1.5 and g > b * 1.5:
                # 对于深绿色，使用黑色和黄色的抖动组合
                if g < 200:
                    return "绿色_深", None
                else:
                    # 浅绿色使用黄色和白色的抖动组合
                    return "绿色_浅", None
            
            # 检测青色(蓝绿色) - 蓝色和绿色分量接近且都高于红色
            if g > 150 and b > 150 and g > r * 1.5 and b > r * 1.5:
                return "青色", None
                
            # 默认情况：使用欧氏距离找最近的标准颜色
            distances = {name: np.sqrt(np.sum((pixel - target_color) ** 2)) 
                        for name, target_color in target_colors.items()}
            closest_color_name = min(distances, key=distances.get)
            return closest_color_name, target_colors[closest_color_name]
        
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
        
        # ==== 方案二: 定义蓝色和绿色的特殊抖动模式 ====
        # 定义蓝色和绿色的特殊抖动模式
        # 使用np.array并确保颜色与目标颜色完全相同
        black = target_colors['黑色'].copy()  # 使用精确的颜色值
        white = target_colors['白色'].copy()
        red = target_colors['红色'].copy()
        yellow = target_colors['黄色'].copy()
        
        # 确保这些颜色是精确的uint8类型
        assert black.dtype == np.uint8
        assert white.dtype == np.uint8
        assert red.dtype == np.uint8
        assert yellow.dtype == np.uint8
        
        blue_dither_patterns = {
            # 深蓝色使用黑色和黄色的交错模式
            "蓝色_深": {
                'pattern': np.array([
                    [black, yellow, black, yellow],
                    [yellow, black, yellow, black],
                    [black, yellow, black, yellow],
                    [yellow, black, yellow, black]
                ]),
                'size': 4
            },
            # 浅蓝色使用白色和黄色的交错模式
            "蓝色_浅": {
                'pattern': np.array([
                    [white, yellow, white, yellow],
                    [yellow, white, yellow, white],
                    [white, yellow, white, yellow],
                    [yellow, white, yellow, white]
                ]),
                'size': 4
            }
        }

        green_dither_patterns = {
            # 深绿色使用黑色和黄色的不同交错模式
            "绿色_深": {
                'pattern': np.array([
                    [yellow, yellow, black, black],
                    [yellow, yellow, black, black],
                    [black, black, yellow, yellow],
                    [black, black, yellow, yellow]
                ]),
                'size': 4
            },
            # 浅绿色使用白色和黄色的不同交错模式
            "绿色_浅": {
                'pattern': np.array([
                    [yellow, yellow, white, white],
                    [yellow, yellow, white, white],
                    [white, white, yellow, yellow],
                    [white, white, yellow, yellow]
                ]),
                'size': 4
            }
        }

        cyan_dither_pattern = {
            # 青色(蓝绿色)使用黄色、白色和黑色的混合模式
            "青色": {
                'pattern': np.array([
                    [yellow, white, yellow, white],
                    [white, black, white, black],
                    [yellow, white, yellow, white],
                    [white, black, white, black]
                ]),
                'size': 4
            }
        }

        # 合并所有特殊颜色的抖动模式
        special_color_patterns = {}
        special_color_patterns.update(blue_dither_patterns)
        special_color_patterns.update(green_dither_patterns)
        special_color_patterns.update(cyan_dither_pattern)
        
        # 保留原有的红色增强处理代码
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
                
                # ==== 方案三: 在抖动处理主循环中检查和处理特殊颜色 ====
                # 检查是否是需要特殊处理的蓝色或绿色
                color_type, mapped_color = map_blue_green_colors(old_pixel, target_colors)
                
                # 如果是特殊颜色类型(蓝色或绿色)，应用特殊抖动模式
                if color_type in special_color_patterns:
                    pattern = special_color_patterns[color_type]['pattern']
                    size = special_color_patterns[color_type]['size']
                    pattern_y = y % size
                    pattern_x = x % size
                    
                    # 获取模式颜色并标准化
                    pattern_color = pattern[pattern_y, pattern_x]
                    standardized_color = standardize_color(pattern_color, target_colors)
                    dithered_image[y, x] = standardized_color
                    
                    processed_pixels += 1
                    continue  # 处理下一个像素
                
                # 原有的颜色处理逻辑
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
                
                # 设置新像素颜色并确保它是精确的目标颜色
                dithered_image[y, x] = standardize_color(new_pixel, target_colors)
                
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
        
        # 将浮点数转换回uint8并强制标准化所有颜色
        dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)
        
        # 在抖动完成后，添加全图颜色校验
        # 严格确保所有像素都是目标颜色之一
        for y in range(h):
            for x in range(w):
                # 强制标准化每一个像素为精确的目标颜色
                dithered_image[y, x] = standardize_color(dithered_image[y, x], target_colors)
        
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
        
        # 在裁剪完成后，添加严格的颜色标准化处理
        # 最终确认：确保裁剪后的图像只包含精确的目标颜色
        for y in range(cropped_image.shape[0]):
            for x in range(cropped_image.shape[1]):
                # 应用严格的颜色标准化
                cropped_image[y, x] = standardize_color(cropped_image[y, x], target_colors)
        
        # 保存前最后一次检查
        target_color_tuples = [tuple(c) for c in target_colors.values()]
        color_count = {c: 0 for c in target_color_tuples}
        non_standard_colors = []
        
        for y in range(cropped_image.shape[0]):
            for x in range(cropped_image.shape[1]):
                pixel = tuple(cropped_image[y, x])
                if pixel in target_color_tuples:
                    color_count[pixel] += 1
                else:
                    non_standard_colors.append(pixel)
                    # 强制修正为黑色
                    cropped_image[y, x] = target_colors['黑色']
        
        # 如果发现非标准颜色，输出警告
        if non_standard_colors:
            print(f"警告：发现{len(non_standard_colors)}个非标准颜色像素，已修正为标准颜色")
        
        # 转换为BGR格式并保存
        cv2.imwrite(output_image_path, cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        
        # --- 5. 将裁剪后的图像转换为墨水屏数据包 ---
        
        # 墨水屏像素定义：
        # 黑 00
        # 白 01
        # 黄 10
        # 红 11
        
        # 创建一个数组来存储所有像素的二进制表示
        binary_pixels = []
        
        # 使用精确的目标颜色元组和它们对应的二进制编码
        standard_colors = {
            tuple(target_colors['黑色']): '00',
            tuple(target_colors['白色']): '01',
            tuple(target_colors['黄色']): '10',
            tuple(target_colors['红色']): '11'
        }
        
        for y in range(200):
            for x in range(200):
                pixel = tuple(cropped_image[y, x])
                
                # 使用标准化的颜色映射
                if pixel in standard_colors:
                    binary_pixels.append(standard_colors[pixel])
                else:
                    # 如果仍然有非标准颜色，进行一次最终修复（虽然这种情况不应该发生）
                    # 找出最接近的标准颜色
                    min_distance = float('inf')
                    closest_color_tuple = tuple(target_colors['黑色'])  # 默认黑色
                    
                    for std_color in standard_colors.keys():
                        distance = sum((pixel[i] - std_color[i])**2 for i in range(3))
                        if distance < min_distance:
                            min_distance = distance
                            closest_color_tuple = std_color
                    
                    binary_pixels.append(standard_colors[closest_color_tuple])
                    # 同时修正像素值
                    cropped_image[y, x] = np.array(closest_color_tuple)
                    print(f"警告：发现遗漏的非标准颜色 {pixel}，已修正为 {closest_color_tuple}")
        
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


def process_hex_data_C(file_path):
    """
    处理十六进制数据文件，提取数据包信息
    
    参数:
        file_path (str): 十六进制数据文件路径
        
    返回:
        list: 提取的十六进制数据包列表
    """
    hex_packets_C = []
    
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
                    hex_packets_C.append(hex_data)
                i += 2  # 跳过数据行
            else:
                i += 1
        else:
            i += 1
    
    return hex_packets_C


def process_hex_data_combined(file_paths_dict):
    """
    处理多个十六进制数据文件，并将结果整合为一个列表
    
    参数:
        file_paths_dict (dict): 包含算法类型和对应文件路径的字典，格式如：
                               {'default': ['path/to/img1.txt', 'path/to/img2.txt'],
                                'T': ['path/to/img1.txt', 'path/to/img2.txt'],
                                'C': ['path/to/img1.txt', 'path/to/img2.txt'],
                                'light_red': ['path/to/img1.txt', 'path/to/img2.txt']}
        
    返回:
        list: 包含算法类型、文件路径和对应二进制数据的字典列表
              格式如：[
                      {'default': '/xxx/img1.png', 'data': ['二进制数据1', '二进制数据2']},
                      {'T': '/xxx/img2.png', 'data': ['二进制数据1', '二进制数据2']},
                      {'C': '/xxx/img3.png', 'data': ['二进制数据1', '二进制数据2']},
                      {'light_red': '/xxx/img4.png', 'data': ['二进制数据1', '二进制数据2']}
                     ]
    """
    result_list = []
    
    # 处理default算法的文件
    if 'default' in file_paths_dict:
        for file_path in file_paths_dict['default']:
            # 从文件路径中提取图片名称并构建完整的图片路径
            img_name = os.path.basename(file_path).replace('default_data.txt', 'default.png')
            img_dir = os.path.dirname(file_path)
            img_path = os.path.normpath(os.path.join(img_dir, img_name))
            # 处理文件并获取二进制数据
            hex_data = process_hex_data_default(file_path)
            # 将结果添加到列表中
            result_list.append({'default': img_path, 'data': hex_data})
    
    # 处理T算法的文件
    if 'T' in file_paths_dict:
        for file_path in file_paths_dict['T']:
            # 从文件路径中提取图片名称并构建完整的图片路径
            img_name = os.path.basename(file_path).replace('T_data.txt', 'T.png')
            img_dir = os.path.dirname(file_path)
            img_path = os.path.normpath(os.path.join(img_dir, img_name))
            # 处理文件并获取二进制数据
            hex_data = process_hex_data_T(file_path)
            # 将结果添加到列表中
            result_list.append({'T': img_path, 'data': hex_data})
    
    # 处理C算法的文件
    if 'C' in file_paths_dict:
        for file_path in file_paths_dict['C']:
            # 从文件路径中提取图片名称并构建完整的图片路径
            img_name = os.path.basename(file_path).replace('C_data.txt', 'C.png')
            img_dir = os.path.dirname(file_path)
            img_path = os.path.normpath(os.path.join(img_dir, img_name))
            # 处理文件并获取二进制数据
            hex_data = process_hex_data_C(file_path)
            # 将结果添加到列表中
            result_list.append({'C': img_path, 'data': hex_data})
    
    # 处理light_red算法的文件
    if 'light_red' in file_paths_dict:
        for file_path in file_paths_dict['light_red']:
            # 从文件路径中提取图片名称并构建完整的图片路径
            img_name = os.path.basename(file_path).replace('light_red_data.txt', 'light_red.png')
            img_dir = os.path.dirname(file_path)
            img_path = os.path.normpath(os.path.join(img_dir, img_name))
            # 处理文件并获取二进制数据
            hex_data = process_hex_data_light_red(file_path)
            # 将结果添加到列表中
            result_list.append({'light_red': img_path, 'data': hex_data})
    
    # 打印结果列表，用于调试
    for result in result_list:
        for algo, path in result.items():
            if algo != 'data':
                print(f"算法: {algo}, 图片路径: {path}")
                print(f"数据包数量: {len(result['data'])}")
    
    return result_list


def process_image_for_eink_light_red(input_path, output_image_path, output_packets_path=""):
    """
    处理图片为四色墨水屏格式并输出相应数据，针对红色像素进行特殊处理，使红色部分看起来比较浅
    
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
        
        # --- 3. 抖动算法处理 ---
        
        # 将目标颜色转换为数组形式，便于计算
        target_colors_array = np.array(list(target_colors.values()))
        
        # 创建一个新的图像用于抖动处理
        dithered_image = scaled_image.copy().astype(np.float32)
        
        # 图像预处理 - 使用双边滤波进行边缘保持平滑
        dithered_image = cv2.bilateralFilter(dithered_image, d=5, sigmaColor=75, sigmaSpace=75)
        
        # 定义红色区域的特殊处理 - 与其他算法不同，这里我们降低红色的强度，使其看起来更浅
        for y in range(h):
            for x in range(w):
                pixel = dithered_image[y, x]
                # 检测红色区域 - 红色分量明显高于其他分量
                if pixel[0] > 170 and pixel[0] > pixel[1] * 1.5 and pixel[0] > pixel[2] * 1.5:
                    # 降低红色强度，但保持适中的红色显示效果
                    # 这与其他算法相反，其他算法是增强红色
                    dithered_image[y, x, 0] = max(160, pixel[0] * 0.9)  # 降低红色分量但保持较高强度
                    dithered_image[y, x, 1] = min(255, pixel[1] * 1.1)  # 轻微增加绿色分量
                    dithered_image[y, x, 2] = min(255, pixel[2] * 1.1)  # 轻微增加蓝色分量
        
        # 定义红色抖动模式 - 使用红色为主、白色为辅的交错模式来表现中等强度的红色
        red = target_colors['红色'].copy()
        white = target_colors['白色'].copy()
        
        red_dither_pattern = np.array([
            [red, red, red, white],
            [red, white, red, red],
            [red, red, red, white],
            [red, white, red, red]
        ])
        
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
            '红色': 25,  # 增加红色阈值，使其更容易被识别为红色
            '黄色': 15
        }
        
        # 定义红色区域的特殊判断参数 - 使用适中的阈值
        red_intensity_threshold = 180  # 降低红色分量强度阈值，使更多区域被识别为红色
        red_ratio_threshold = 1.7     # 降低红色与其他颜色分量的比例阈值
        
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
                    # 对于红色，即使足够接近，也使用抖动模式而不是纯色
                    if closest_color_name == '红色':
                        # 使用红色抖动模式
                        pattern_y = y % 4
                        pattern_x = x % 4
                        dithered_image[y, x] = red_dither_pattern[pattern_y, pattern_x]
                        is_close_to_target = True
                    else:
                        is_close_to_target = True
                        dithered_image[y, x] = closest_target_color  # 直接设置为目标颜色
                    
                # 特殊处理红色区域：使用定义的参数判断是否为红色区域
                elif closest_color_name == '红色' and old_pixel[0] > red_intensity_threshold and old_pixel[0] > old_pixel[1] * red_ratio_threshold and old_pixel[0] > old_pixel[2] * red_ratio_threshold:
                    # 使用红色抖动模式而不是纯红色
                    pattern_y = y % 4
                    pattern_x = x % 4
                    dithered_image[y, x] = red_dither_pattern[pattern_y, pattern_x]
                    is_close_to_target = True
                
                # 如果颜色足够接近目标色，则跳过误差扩散
                if is_close_to_target:
                    processed_pixels += 1
                    continue  # 处理下一个像素
                
                # 如果颜色不够接近任何目标色，则执行标准抖动
                # 找到最接近的目标颜色
                distances = np.sqrt(np.sum((old_pixel[np.newaxis, :] - target_colors_array) ** 2, axis=1))
                closest_color_index = np.argmin(distances)
                new_pixel = target_colors_array[closest_color_index]
                
                # 对于接近红色但不够接近的像素，使用抖动模式
                if tuple(new_pixel) == tuple(target_colors['红色']):
                    pattern_y = y % 4
                    pattern_x = x % 4
                    new_pixel = red_dither_pattern[pattern_y, pattern_x]
                
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


def process_hex_data_light_red(file_path):
    """
    处理十六进制数据文件，提取数据包信息
    
    参数:
        file_path (str): 十六进制数据文件路径
        
    返回:
        list: 提取的十六进制数据包列表
    """
    hex_packets_light_red = []
    
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
                    hex_packets_light_red.append(hex_data)
                i += 2  # 跳过数据行
            else:
                i += 1
        else:
            i += 1
    
    return hex_packets_light_red


def process_image_combined(image_name, target_dir):
    """
    封装图像处理功能，使用四种算法处理图像并返回整合的数据
    
    参数:
        image_name (str): 图像名称（可以包含或不包含扩展名）
        target_dir (str): 目标文件夹路径
    
    返回:
        list: process_hex_data_combined函数整合的结果列表
    """
    print(f"====== 开始处理图像: {image_name} ======")
    
    # 从image_name中提取基本名称（不含扩展名）和扩展名
    base_name, ext = os.path.splitext(image_name)
    
    # 如果没有提供扩展名，尝试常见的图像扩展名
    if not ext:
        # 按优先级尝试不同的扩展名
        possible_extensions = ['.jpg', '.png', '.jpeg', '.bmp']
        found_extension = None
        
        for extension in possible_extensions:
            if os.path.exists(f"{target_dir}/{image_name}{extension}"):
                found_extension = extension
                break
        
        if found_extension:
            # 找到有效的扩展名，使用它
            ext = found_extension
            input_image = f"{target_dir}/{image_name}{ext}"
            print(f"自动检测到图像扩展名: {ext}")
        else:
            # 未找到文件，默认使用jpg
            ext = '.jpg'
            input_image = f"{target_dir}/{image_name}{ext}"
            print(f"未检测到图像文件，默认使用扩展名: {ext}")
    else:
        # 已提供扩展名，直接使用
        input_image = f"{target_dir}/{image_name}"
        print(f"使用提供的图像路径: {input_image}")
        
        # 更新base_name以确保文件夹名称不包含扩展名
        image_name = base_name
    
    # 拼接文件夹路径 - 使用不含扩展名的基本名称
    folder_path = f"{target_dir}/{base_name}"
    print(f"图像文件夹路径: {folder_path}")
    
    # 如果文件夹不存在则创建
    os.makedirs(folder_path, exist_ok=True)
    print(f"已确保文件夹存在: {folder_path}")
    
    # 定义输入和输出路径（input_image已在上面定义）
    print(f"输入图像路径: {input_image}")
    
    # 定义四种算法的输出图像路径
    output_image_default = f"{folder_path}/{base_name}_default.png"
    output_image_T = f"{folder_path}/{base_name}_T.png"
    output_image_C = f"{folder_path}/{base_name}_C.png"
    output_image_light_red = f"{folder_path}/{base_name}_light_red.png"
    print(f"输出图像路径 - default: {output_image_default}")
    print(f"输出图像路径 - T: {output_image_T}")
    print(f"输出图像路径 - C: {output_image_C}")
    print(f"输出图像路径 - light_red: {output_image_light_red}")
    
    # 定义四种算法的输出数据包路径
    output_data_default = f"{folder_path}/default_data.txt"
    output_data_T = f"{folder_path}/T_data.txt"
    output_data_C = f"{folder_path}/C_data.txt"
    output_data_light_red = f"{folder_path}/light_red_data.txt"
    print(f"输出数据路径 - default: {output_data_default}")
    print(f"输出数据路径 - T: {output_data_T}")
    print(f"输出数据路径 - C: {output_data_C}")
    print(f"输出数据路径 - light_red: {output_data_light_red}")
    
    # 处理图像并生成数据包
    try:
        # 检查输入图像是否存在
        if not os.path.exists(input_image):
            print(f"错误: 输入图像不存在: {input_image}")
            return []
        
        print(f"\n----- 开始使用default算法处理图像 -----")
        # 使用default算法处理图像
        result_default = process_image_for_eink_default(input_image, output_image_default, output_data_default)
        print(f"default算法处理结果: {'成功' if result_default else '失败'}")
        
        print(f"\n----- 开始使用T算法处理图像 -----")
        # 使用T算法处理图像
        result_T = process_image_for_eink_T(input_image, output_image_T, output_data_T)
        print(f"T算法处理结果: {'成功' if result_T else '失败'}")
        
        print(f"\n----- 开始使用C算法处理图像 -----")
        # 使用C算法处理图像
        result_C = process_image_for_eink_C(input_image, output_image_C, output_data_C)
        print(f"C算法处理结果: {'成功' if result_C else '失败'}")
        
        print(f"\n----- 开始使用light_red算法处理图像 -----")
        # 使用light_red算法处理图像
        result_light_red = process_image_for_eink_light_red(input_image, output_image_light_red, output_data_light_red)
        print(f"light_red算法处理结果: {'成功' if result_light_red else '失败'}")
        
        print(f"\n----- 开始整合处理结果 -----")
        # 整合处理结果
        file_paths_dict = {
            'default': [output_data_default],
            'T': [output_data_T],
            'C': [output_data_C],
            'light_red': [output_data_light_red]
        }
        
        # 检查数据文件是否存在
        all_files_exist = True
        for algo, paths in file_paths_dict.items():
            for path in paths:
                if not os.path.exists(path):
                    print(f"警告: {algo}算法的数据文件不存在: {path}")
                    all_files_exist = False
        
        if not all_files_exist:
            print("部分数据文件不存在，可能影响整合结果")
            
        # 调用整合函数获取结果
        combined_results = process_hex_data_combined(file_paths_dict)
        print(f"整合结果包含 {len(combined_results)} 个条目")
        
        # 输出处理结果
        if result_default and result_T and result_C and result_light_red:
            print(f"图片 {base_name} 处理成功，生成了四种算法的处理结果")
        else:
            print(f"图片 {base_name} 处理部分失败，请检查输入路径和图像文件")
        
        print(f"====== 完成处理图像: {base_name} ======\n")
        return combined_results
    
    except Exception as e:
        import traceback
        print(f"处理图像时出现异常: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return []

# 示例用法
if __name__ == "__main__":
    # 示例：处理一张图片
    imageName = "Maggie-1"    
    targetDir = f"/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/Pic_Color_Select【成品】/1/pic"
    
    # 调用封装函数处理图片
    combined_results = process_image_combined(imageName, targetDir)
    
    if combined_results:
        print(f"整合后的结果包含 {len(combined_results)} 个图片的数据")
    else:
        print("图片处理失败，没有生成有效的结果数据。")