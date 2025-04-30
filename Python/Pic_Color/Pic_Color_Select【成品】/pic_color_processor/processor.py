# -*- coding: utf-8 -*-
"""
Processor Module
===============

这个模块包含图像处理和墨水屏数据生成的核心类。

主要类:
    - PicColorProcessor: 主处理器类，整合了所有功能
    - ImageProcessor: 处理图像缩放和裁剪
    - ColorMapper: 处理颜色映射
    - DitherProcessor: 实现抖动算法
    - EInkPacketGenerator: 生成墨水屏数据包
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans


class ImageProcessor:
    """
    处理图像缩放和裁剪的类
    """
    def __init__(self):
        pass
        
    def resize_image(self, image, resize_to=200):
        """
        按比例缩放图像，保持宽高比
        
        Args:
            image: 输入图像
            resize_to: 目标尺寸
            
        Returns:
            缩放后的图像
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
        return scaled_image
    
    def crop_image(self, image, target_size=200, center=True):
        """
        裁剪图像到目标尺寸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸
            center: 是否从中心裁剪
            
        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]
        
        if w == target_size and h == target_size:
            return image
            
        if center:
            center_x, center_y = w // 2, h // 2
            start_x = max(0, center_x - target_size // 2)
            start_y = max(0, center_y - target_size // 2)
            if start_x + target_size > w:
                start_x = w - target_size
            if start_y + target_size > h:
                start_y = h - target_size
        else:
            start_x, start_y = 0, 0
            
        cropped_image = image[start_y:start_y+target_size, start_x:start_x+target_size]
        return cropped_image


class ColorMapper:
    """
    处理颜色映射的类
    """
    def __init__(self, n_colors=10):
        self.n_colors = n_colors
        self.target_colors = {
            '黑色': np.array([0, 0, 0]),
            '白色': np.array([255, 255, 255]),
            '红色': np.array([255, 0, 0]),
            '黄色': np.array([255, 255, 0])
        }
        self.rgb_to_color_name = {tuple(rgb): name for name, rgb in self.target_colors.items()}
        
    def cluster_colors(self, image):
        """
        使用K-means聚类分析图像颜色
        
        Args:
            image: 输入图像
            
        Returns:
            聚类后的调色板
        """
        h, w, d = image.shape
        pixels = image.reshape(h * w, d)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42, n_init='auto', verbose=0).fit(pixels)
        palette = np.uint8(kmeans.cluster_centers_)
        return palette
        
    def map_colors(self, palette):
        """
        将颜色映射到目标颜色
        
        Args:
            palette: 颜色调色板
            
        Returns:
            映射后的调色板
        """
        mapped_palette = []
        for color in palette:
            distances = {name: np.sqrt(np.sum((color - target_color) ** 2)) 
                        for name, target_color in self.target_colors.items()}
            closest_color_name = min(distances, key=distances.get)
            closest_color = self.target_colors[closest_color_name]
            mapped_palette.append(closest_color)
        return np.array(mapped_palette)


class DitherProcessor:
    """
    实现抖动算法的类
    """
    def __init__(self, target_colors=None):
        if target_colors is None:
            self.target_colors = {
                '黑色': np.array([0, 0, 0]),
                '白色': np.array([255, 255, 255]),
                '红色': np.array([255, 0, 0]),
                '黄色': np.array([255, 255, 0])
            }
        else:
            self.target_colors = target_colors
            
    def apply_dithering(self, image):
        """
        应用抖动算法处理图像
        
        Args:
            image: 输入图像
            
        Returns:
            抖动处理后的图像
        """
        h, w = image.shape[:2]
        target_colors_array = np.array(list(self.target_colors.values()))
        dithered_image = image.copy().astype(np.float32)
        dithered_image = cv2.bilateralFilter(dithered_image, d=5, sigmaColor=75, sigmaSpace=75)
        
        # Jarvis, Judice, and Ninke抖动算法权重
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
        return dithered_image


class EInkPacketGenerator:
    """
    生成墨水屏数据包的类
    """
    def __init__(self):
        self.color_to_binary = {
            '黑色': '00',
            '白色': '01',
            '黄色': '10',
            '红色': '11'
        }
        self.target_colors = {
            '黑色': np.array([0, 0, 0]),
            '白色': np.array([255, 255, 255]),
            '红色': np.array([255, 0, 0]),
            '黄色': np.array([255, 255, 0])
        }
        self.rgb_to_color_name = {tuple(rgb): name for name, rgb in self.target_colors.items()}
        
    def generate_packets(self, image):
        """
        从图像生成墨水屏数据包
        
        Args:
            image: 处理后的图像，应为200x200像素
            
        Returns:
            数据包列表
        """
        if image.shape[:2] != (200, 200):
            raise ValueError(f"图像尺寸必须为200x200像素，当前为{image.shape[:2]}")
            
        binary_pixels = []
        for y in range(200):
            for x in range(200):
                pixel = tuple(image[y, x])
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
            packets.append(complete_packet)
            
        # 添加最后一个包，表示屏幕更新
        last_packet_number = 0x32
        last_packet_data = [ord(c) for c in "screen updating\r\n"]
        last_packet_checksum = (last_packet_number + sum(last_packet_data)) % 256
        last_packet = [last_packet_number] + last_packet_data + [last_packet_checksum]
        packets.append(last_packet)
        
        return packets


class PicColorProcessor:
    """
    主处理器类，整合了所有功能
    """
    def __init__(self, n_colors=10):
        self.image_processor = ImageProcessor()
        self.color_mapper = ColorMapper(n_colors=n_colors)
        self.dither_processor = DitherProcessor()
        self.eink_generator = EInkPacketGenerator()
        
    def process_image(self, image, resize_to=200, crop_center=True):
        """
        处理图像，包括缩放、颜色聚类、颜色映射、抖动处理和裁剪
        
        Args:
            image: 输入图像
            resize_to: 目标尺寸
            crop_center: 是否从中心裁剪
            
        Returns:
            处理后的图像
        """
        # 缩放图像
        scaled_image = self.image_processor.resize_image(image, resize_to=resize_to)
        
        # 应用抖动算法
        dithered_image = self.dither_processor.apply_dithering(scaled_image)
        
        # 裁剪图像
        cropped_image = self.image_processor.crop_image(dithered_image, target_size=resize_to, center=crop_center)
        
        return cropped_image
    
    def image_to_eink_packets(self, image):
        """
        从图像生成墨水屏数据包
        
        Args:
            image: 处理后的图像
            
        Returns:
            数据包列表
        """
        return self.eink_generator.generate_packets(image)