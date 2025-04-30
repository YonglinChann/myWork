import numpy as np
import cv2
from sklearn.cluster import KMeans

class PicColorProcessor:
    def __init__(self, n_colors=10):
        self.n_colors = n_colors
        self.target_colors = {
            '黑色': np.array([0, 0, 0]),
            '白色': np.array([255, 255, 255]),
            '红色': np.array([255, 0, 0]),
            '黄色': np.array([255, 255, 0])
        }
        self.color_to_binary = {
            '黑色': '00',
            '白色': '01',
            '黄色': '10',
            '红色': '11'
        }
        self.rgb_to_color_name = {tuple(rgb): name for name, rgb in self.target_colors.items()}

    def process_image(self, image, resize_to=200, crop_center=True):
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
            packets.append(complete_packet)
        last_packet_number = 0x32
        last_packet_data = [ord(c) for c in "screen updating\r\n"]
        last_packet_checksum = (last_packet_number + sum(last_packet_data)) % 256
        last_packet = [last_packet_number] + last_packet_data + [last_packet_checksum]
        packets.append(last_packet)
        return packets