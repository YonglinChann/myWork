# -*- coding: utf-8 -*-
"""
Pic Color Processor
==================

一个用于图像处理和墨水屏数据生成的Python库。

这个库提供了以下功能：
1. 图像缩放和裁剪
2. K-means颜色聚类分析
3. 颜色映射到目标颜色（黑、白、红、黄）
4. 抖动算法处理
5. 墨水屏数据包生成

基本用法:
    >>> from pic_color_processor import PicColorProcessor
    >>> import cv2
    >>> # 读取图像
    >>> image = cv2.imread('example.jpg')
    >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    >>> # 创建处理器实例
    >>> processor = PicColorProcessor()
    >>> # 处理图像
    >>> processed_image = processor.process_image(image)
    >>> # 生成墨水屏数据包
    >>> packets = processor.image_to_eink_packets(processed_image)
"""

from .processor import PicColorProcessor, ImageProcessor, ColorMapper, DitherProcessor, EInkPacketGenerator

__version__ = '0.1.0'
__all__ = ['PicColorProcessor', 'ImageProcessor', 'ColorMapper', 'DitherProcessor', 'EInkPacketGenerator']