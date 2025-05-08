# pic_processor.py
# 图像处理器模块，用于处理图像并生成电子墨水屏幕数据包

import numpy as np
import cv2
from sklearn.cluster import KMeans
import os

class PicProcessor:
    """
    图像处理器类，用于处理图像并生成适用于电子墨水屏幕的数据包
    
    该类提供了图像处理功能，包括缩放、颜色聚类、抖动处理和数据包生成。
    专注于单图片处理，简化接口设计。
    """
    
    def __init__(self, n_colors=10):
        """
        初始化图像处理器
        
        参数:
            n_colors (int): 颜色聚类的数量，默认为10
        """
        self.n_colors = n_colors
            
        # 目标颜色定义
        self.target_colors = {
            '黑色': np.array([0, 0, 0]),
            '白色': np.array([255, 255, 255]),
            '红色': np.array([255, 0, 0]),
            '黄色': np.array([255, 255, 0])
        }
        
        # 颜色到二进制的映射
        self.color_to_binary = {
            '黑色': '00',
            '白色': '01',
            '黄色': '10',
            '红色': '11'
        }
        
        # RGB值到颜色名称的映射
        self.rgb_to_color_name = {tuple(rgb): name for name, rgb in self.target_colors.items()}
    
    def load_image(self, image_path):
        """
        加载图像文件
        
        参数:
            image_path (str): 图像文件路径
            
        返回:
            numpy.ndarray: 加载的图像数据
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像文件: {image_path}")
            
        return image
    
    def process_image(self, image, resize_to=200, crop_center=True):
        """
        处理图像，包括缩放、颜色聚类和抖动处理
        
        参数:
            image (numpy.ndarray): 输入图像数据
            resize_to (int): 缩放尺寸，默认为200
            crop_center (bool): 是否从中心裁剪，默认为True
            
        返回:
            numpy.ndarray: 处理后的图像数据
        """
        h, w, d = image.shape
        if h < resize_to or w < resize_to:
            raise ValueError(f"图片尺寸({w}x{h})小于所需的{resize_to}x{resize_to}像素")
            
        # 缩放
        if w < h:
            scale_factor = resize_to / w
            scaled_w = resize_to
            scaled_h = int(h * scale_factor)
        else:
            scale_factor = resize_to / h
            scaled_h = resize_to
            scaled_w = int(w * scale_factor)
            
        scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        h, w, d = scaled_image.shape
        
        # KMeans聚类
        pixels = scaled_image.reshape(h * w, d)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init='auto', verbose=0).fit(pixels)
        palette = np.uint8(kmeans.cluster_centers_)
        
        # 颜色映射
        mapped_palette = []
        for color in palette:
            distances = {name: np.sqrt(np.sum((color - target_color) ** 2)) for name, target_color in self.target_colors.items()}
            closest_color_name = min(distances, key=distances.get)
            closest_color = self.target_colors[closest_color_name]
            mapped_palette.append(closest_color)
        mapped_palette = np.array(mapped_palette)
        
        # 抖动算法
        target_colors_array = np.array(list(self.target_colors.values()))
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
        
        # Jarvis, Judice, and Ninke 抖动权重
        jjn_weights = np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1]
        ]) / 48
        
        # 应用改进的抖动算法
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
                for color_name, target_color in self.target_colors.items():
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
                    dithered_image[y, x] = self.target_colors['红色']  # 强制设为红色
                    
                # 额外检查：如果红色分量非常高，即使不是最接近的颜色也强制设为红色
                elif old_pixel[0] > 220 and old_pixel[0] > old_pixel[1] * 2.0 and old_pixel[0] > old_pixel[2] * 2.0:
                    is_close_to_target = True
                    dithered_image[y, x] = self.target_colors['红色']  # 强制设为红色
                
                # 如果颜色足够接近目标色，则跳过误差扩散
                if is_close_to_target:
                    continue  # 处理下一个像素
                
                # 如果颜色不够接近任何目标色，则执行标准抖动
                distances = np.sqrt(np.sum((old_pixel - target_colors_array) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                new_pixel = target_colors_array[closest_idx]
                dithered_image[y, x] = new_pixel
                
                # 计算量化误差
                quant_error = old_pixel - new_pixel
                
                # 扩散误差到邻近像素
                for dy in range(3):
                    for dx in range(5):
                        if jjn_weights[dy, dx] > 0:
                            ny, nx = y + dy, x + dx - 2
                            if 0 <= ny < h and 0 <= nx < w:
                                dithered_image[ny, nx] += quant_error * jjn_weights[dy, dx]
        
        dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)
        
        # 裁剪
        if w == resize_to and h == resize_to:
            cropped_image = dithered_image
        else:
            if crop_center:
                center_x, center_y = w // 2, h // 2
                start_x = max(0, center_x - resize_to // 2)
                start_y = max(0, center_y - resize_to // 2)
                if start_x + resize_to > w:
                    start_x = w - resize_to
                if start_y + resize_to > h:
                    start_y = h - resize_to
            else:
                start_x, start_y = 0, 0
            cropped_image = dithered_image[start_y:start_y+resize_to, start_x:start_x+resize_to]
            
        return cropped_image
    
    def image_to_eink_packets(self, cropped_image):
        """
        将处理后的图像转换为电子墨水屏幕数据包
        
        参数:
            cropped_image (numpy.ndarray): 处理后的图像数据，应为200x200像素
            
        返回:
            list: 数据包字符串列表，每个元素是一个完整的数据包字符串
        """
        binary_pixels = []
        for y in range(200):
            for x in range(200):
                pixel = tuple(cropped_image[y, x])
                color_name = self.rgb_to_color_name[pixel]
                binary_pixels.append(self.color_to_binary[color_name])
        
        byte_data = []
        current_byte = ''
        for binary_pixel in binary_pixels:
            current_byte += binary_pixel
            if len(current_byte) == 8:
                byte_value = int(current_byte, 2)
                byte_data.append(byte_value)
                current_byte = ''
        
        if current_byte:
            current_byte = current_byte.ljust(8, '0')
            byte_value = int(current_byte, 2)
            byte_data.append(byte_value)
        
        packets = []
        for i in range(50):
            packet_number = i
            packet_data = byte_data[i * 200:(i + 1) * 200]
            checksum = (packet_number + sum(packet_data)) % 256
            complete_packet = [packet_number] + packet_data + [checksum]
            # 将数值列表转换为单个字符串
            packet_str = ''.join(str(value) for value in complete_packet)
            packets.append(packet_str)
        
        # 添加最后一个更新屏幕的数据包
        last_packet_number = 0x32
        last_packet_data = [ord(c) for c in "screen updating\r\n"]
        last_packet_checksum = (last_packet_number + sum(last_packet_data)) % 256
        last_packet = [last_packet_number] + last_packet_data + [last_packet_checksum]
        # 将最后一个数据包也转换为字符串
        last_packet_str = ''.join(str(value) for value in last_packet)
        packets.append(last_packet_str)
        
        return packets
    
    def save_processed_image(self, image, output_path):
        """
        保存处理后的图像
        
        参数:
            image (numpy.ndarray): 图像数据
            output_path (str): 输出文件的完整路径
            
        返回:
            str: 保存的文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cv2.imwrite(output_path, image)
        return output_path
    
    def save_eink_packets(self, packets, output_path):
        """
        保存电子墨水屏幕数据包
        
        参数:
            packets (list): 数据包字符串列表
            output_path (str): 输出文件的完整路径
            
        返回:
            str: 保存的文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_path, 'w') as f:
            for packet in packets:
                f.write(packet + '\n')
                
        return output_path
    
    def process_file(self, input_path, output_image_path=None, output_packets_path=None, resize_to=200, crop_center=True):
        """
        处理单个图像文件
        
        参数:
            input_path (str): 输入图像文件的完整路径
            output_image_path (str): 输出处理后图像的完整路径，如果为None则不保存图像
            output_packets_path (str): 输出数据包的完整路径，如果为None则不保存数据包
            resize_to (int): 缩放尺寸，默认为200
            crop_center (bool): 是否从中心裁剪，默认为True
            
        返回:
            tuple: (处理后的图像, 数据包列表, 图像保存路径, 数据包保存路径)
        """
        # 加载图像
        image = self.load_image(input_path)
        
        # 处理图像
        processed_image = self.process_image(image, resize_to, crop_center)
        
        # 生成数据包
        packets = self.image_to_eink_packets(processed_image)
        
        # 保存处理后的图像
        image_save_path = None
        if output_image_path:
            image_save_path = self.save_processed_image(processed_image, output_image_path)
        
        # 保存数据包
        packets_save_path = None
        if output_packets_path:
            packets_save_path = self.save_eink_packets(packets, output_packets_path)
        
        return processed_image, packets, image_save_path, packets_save_path