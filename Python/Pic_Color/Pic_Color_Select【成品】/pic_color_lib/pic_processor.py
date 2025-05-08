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
        dithered_image = cv2.bilateralFilter(dithered_image, d=5, sigmaColor=75, sigmaSpace=75)
        
        # Jarvis, Judice, and Ninke 抖动权重
        jjn_weights = np.array([
            [0, 0, 0, 7, 5],
            [3, 5, 7, 5, 3],
            [1, 3, 5, 3, 1]
        ]) / 48
        
        for y in range(h):
            for x in range(w):
                old_pixel = dithered_image[y, x].copy()
                local_region = dithered_image[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
                local_mean = np.mean(local_region, axis=(0, 1))
                old_pixel = old_pixel * 0.7 + local_mean * 0.3
                
                distances = np.sqrt(np.sum((old_pixel - target_colors_array) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                new_pixel = target_colors_array[closest_idx]
                dithered_image[y, x] = new_pixel
                
                quant_error = old_pixel - new_pixel
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
    
    def process_file(self, input_path, output_path=None, output_image_path=None, output_packets_path=None, resize_to=200, crop_center=True):
        """
        处理单个图像文件
        
        参数:
            input_path (str): 输入图像文件的完整路径
            output_path (str): 输出文件的基础路径，如果提供，将自动生成图像和数据包文件路径
            output_image_path (str): 输出处理后图像的完整路径，如果为None且output_path不为None，则使用output_path生成
            output_packets_path (str): 输出数据包的完整路径，如果为None且output_path不为None，则使用output_path生成
            resize_to (int): 缩放尺寸，默认为200
            crop_center (bool): 是否从中心裁剪，默认为True
            
        返回:
            tuple: (处理后的图像, 数据包列表, 图像保存路径, 数据包保存路径)
                  - 处理后的图像: numpy.ndarray 类型，处理后的图像数据
                  - 数据包列表: list 类型，包含所有生成的数据包字符串，可直接用于传输
                  - 图像保存路径: str 类型，处理后图像的保存路径，如未保存则为None
                  - 数据包保存路径: str 类型，数据包的保存路径，如未保存则为None
        """
        # 处理输出路径
        if output_path is not None:
            # 如果提供了output_path但没有提供特定的输出路径，则自动生成
            if output_image_path is None:
                base_name = os.path.basename(input_path)
                name, _ = os.path.splitext(base_name)
                output_image_path = f"{output_path}/{name}_processed.png"
            
            if output_packets_path is None:
                base_name = os.path.basename(input_path)
                name, _ = os.path.splitext(base_name)
                output_packets_path = f"{output_path}/{name}_packets.txt"
        
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