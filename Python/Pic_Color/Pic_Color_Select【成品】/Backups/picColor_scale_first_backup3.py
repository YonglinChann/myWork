# 通过使用 K-means 聚类算法对图像进行颜色聚类分析，并生成一个基于聚类中心(即最具代表性的颜色)的RGB值和调色板。
# 使用抖动算法对颜色进行处理
# 先缩放图像到200像素，再进行抖动处理
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

# 设置matplotlib中文字体
plt.rcParams['font.family'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建输出目录
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 创建photos文件夹的路径
photos_dir = os.path.join(current_dir, 'photos')

# 检查photos文件夹是否存在
if not os.path.exists(photos_dir):
    os.makedirs(photos_dir)
    print(f"已创建photos文件夹: {photos_dir}")
    print("请将需要分析的图片放入此文件夹，并命名为photo1.jpg、photo2.jpg等格式")
    exit(0)

# 查找photos文件夹中所有photo开头的图片文件
photo_files = []
for ext in ['jpg', 'jpeg', 'png', 'bmp']:
    photo_files.extend(glob.glob(os.path.join(photos_dir, f"photo*[0-9].{ext}")))

if not photo_files:
    raise FileNotFoundError(
        f'未在photos文件夹中找到photo开头的图片文件，请确保在{photos_dir}中存在photo1.jpg或photo2.png等格式的图片')

# 整理图片文件，按照编号排序
photo_dict = {}
for photo_path in photo_files:
    filename = os.path.basename(photo_path)
    # 提取文件名中的数字部分
    import re

    match = re.search(r'photo(\d+)', filename)
    if match:
        photo_num = int(match.group(1))
        photo_dict[photo_num] = photo_path

# 默认使用photo3图片
photo_num = 3
if photo_num in photo_dict:
    photo_path = photo_dict[photo_num]
else:
    # 如果没有photo3，则使用第一张可用的图片
    photo_num = min(photo_dict.keys()) if photo_dict else 0
    if photo_num > 0:
        photo_path = photo_dict[photo_num]
    else:
        raise FileNotFoundError(f'未在photos文件夹中找到可用的图片文件')

print(f"将处理图片: {os.path.basename(photo_path)}")

# 使用OpenCV读取图像（BGR格式）
image = cv2.imread(photo_path)
if image is None:
    raise ValueError(f'无法读取图片文件：{photo_path}')

# 转换为RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像尺寸
h, w, d = image.shape

# 检查图片尺寸是否满足要求 (不小于200*200)
if h < 200 or w < 200:
    raise ValueError(f'处理的图片尺寸({w}x{h})小于所需的200x200像素')

print(f"原始图片尺寸: {w}x{h}像素")

# --- 1. 先进行图像缩放 ---
print("\n开始进行图像缩放...")

# 计算缩放比例，使较短的边等于200像素
if w < h:  # 宽度更小
    scale_factor = 200 / w
    scaled_w = 200
    scaled_h = int(h * scale_factor)
    print(f"以宽度为基准进行缩放，缩放比例: {scale_factor:.2f}")
else:  # 高度更小或相等
    scale_factor = 200 / h
    scaled_h = 200
    scaled_w = int(w * scale_factor)
    print(f"以高度为基准进行缩放，缩放比例: {scale_factor:.2f}")

# 缩放图像
scaled_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
print(f"缩放后的图片尺寸: {scaled_w}x{scaled_h}像素")

# 更新尺寸变量
h, w, d = scaled_image.shape

# --- 2. K-means 颜色聚类分析 ---
# 重塑图像为二维数组，每行代表一个像素的RGB值
pixels = scaled_image.reshape(h * w, d)

# 设置固定的颜色提取数量为10
n_colors = 10

print(f"将使用 {n_colors} 个聚类进行颜色分析")

# K-means聚类，n_init设置为auto
print("\n开始进行颜色聚类分析...")
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto', verbose=1).fit(pixels)
print("颜色聚类分析完成！\n")

# 获取聚类中心（调色板）
palette = np.uint8(kmeans.cluster_centers_)
print("提取的原始颜色：")
print(palette)

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

mapped_palette = np.array(mapped_palette)

# 打印映射结果
print("\n颜色映射结果：")
for i, mapping in color_mapping.items():
    print(
        f"颜色 {i + 1}: RGB{mapping['原始颜色']} -> {mapping['颜色名称']} RGB{mapping['映射颜色']} (距离: {mapping['距离']:.2f})")

# 创建一个展示原始调色板的图像
palette_image = np.zeros((100, n_colors * 50, 3), dtype=np.uint8)
for i in range(n_colors):
    palette_image[:, i * 50:(i + 1) * 50] = palette[i]

# 创建一个展示映射后调色板的图像
mapped_palette_image = np.zeros((100, n_colors * 50, 3), dtype=np.uint8)
for i in range(n_colors):
    mapped_palette_image[:, i * 50:(i + 1) * 50] = mapped_palette[i]

# 显示原始图像、缩放后的图像、原始调色板和映射后的调色板
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(4, 1, 2)
plt.imshow(scaled_image)
plt.title(f'缩放后的图像 ({scaled_w}x{scaled_h})')
plt.axis('off')

plt.subplot(4, 1, 3)
plt.imshow(palette_image)
plt.title('提取的原始颜色调色板')
plt.axis('off')

plt.subplot(4, 1, 4)
plt.imshow(mapped_palette_image)
plt.title('映射到开发板支持颜色的调色板')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'color_palette.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- 3. 抖动算法处理 ---
print("\n开始进行抖动算法处理...")

# 将目标颜色转换为数组形式，便于计算
target_colors_array = np.array(list(target_colors.values()))

# 创建一个新的图像用于抖动处理
dithered_image = scaled_image.copy().astype(np.float32)

# 图像预处理 - 使用双边滤波进行边缘保持平滑
print("正在进行图像预处理...")
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

print("图像预处理完成！")

# 定义Jarvis-Judice-Ninke抖动矩阵权重
jjn_weights = [
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
]
jjn_weights = np.array(jjn_weights) / 48  # 归一化权重

# 定义颜色阈值，用于判断颜色是否足够接近目标颜色
# 为红色区域设置更严格的阈值，以更好地保留纯红色区域
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
print_interval = total_pixels // 100  # 每处理1%的像素打印一次进度

print("\n开始应用抖动算法...")
for y in range(h):
    for x in range(w):
        # 获取当前像素的颜色（已经过预处理）
        old_pixel = dithered_image[y, x].copy()

        # --- 检查是否接近目标颜色，使用针对不同颜色的阈值 ---
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
            dithered_image[y, x] = closest_target_color # 直接设置为目标颜色
            
        # 特殊处理红色区域：使用定义的参数判断是否为红色区域
        elif closest_color_name == '红色' and old_pixel[0] > red_intensity_threshold and old_pixel[0] > old_pixel[1] * red_ratio_threshold and old_pixel[0] > old_pixel[2] * red_ratio_threshold:
            is_close_to_target = True
            dithered_image[y, x] = target_colors['红色'] # 强制设为红色
            
        # 额外检查：如果红色分量非常高，即使不是最接近的颜色也强制设为红色
        elif old_pixel[0] > 220 and old_pixel[0] > old_pixel[1] * 2.0 and old_pixel[0] > old_pixel[2] * 2.0:
            is_close_to_target = True
            dithered_image[y, x] = target_colors['红色'] # 强制设为红色

        # 如果颜色足够接近目标色，则跳过误差扩散
        if is_close_to_target:
            processed_pixels += 1
            if processed_pixels % print_interval == 0:
                progress = (processed_pixels / total_pixels) * 100
                print(f"抖动处理进度: {progress:.1f}%", end='\r')
            continue # 处理下一个像素
        # --- 新增结束 ---

        # 如果颜色不够接近任何目标色，则执行标准抖动
        # 找到最接近的目标颜色 (如果上面没找到足够近的，这里重新计算一次最接近的)
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
        if processed_pixels % print_interval == 0:
            progress = (processed_pixels / total_pixels) * 100
            print(f"抖动处理进度: {progress:.1f}%", end='\r')

print("\n抖动算法处理完成！")

# 将浮点数转换回uint8
dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)

# 显示缩放后的图像和抖动处理后的图像
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.imshow(scaled_image)
plt.title('缩放后的图像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(dithered_image)
plt.title('抖动处理后（四色）')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'processed_image.png'), dpi=300, bbox_inches='tight')
plt.show()

# --- 4. 裁剪图像（如果需要）---
print("\n检查是否需要裁剪图像...")

# 获取抖动后图像的尺寸
h_dith, w_dith, _ = dithered_image.shape

# 判断是否需要裁剪
if w_dith == 200 and h_dith == 200:
    print("图像尺寸已经是200x200像素，无需裁剪")
    cropped_image = dithered_image
else:
    print(f"需要将处理后的图像裁剪为200x200像素")
    print(f"当前尺寸: {w_dith}x{h_dith}像素")

    # 计算图片中心位置
    center_x, center_y = w_dith // 2, h_dith // 2

    # 询问用户是否使用默认中心裁剪
    use_center = input("默认将以图片中心点进行200x200像素的裁剪，是否同意？(y/n): ").lower().strip()

    if use_center == 'y' or use_center == '':
        # 以图片中心点为中心进行裁剪
        start_x = max(0, center_x - 100)
        start_y = max(0, center_y - 100)
        
        # 调整以确保能裁剪到完整的200x200区域
        if start_x + 200 > w_dith:
            start_x = w_dith - 200
        if start_y + 200 > h_dith:
            start_y = h_dith - 200
            
        print(f"将以中心点({center_x}, {center_y})为中心进行裁剪")
    else:
        # 以左下角为原点进行裁剪
        print(f"图片尺寸为{w_dith}x{h_dith}，请指定裁剪起始点坐标（左下角为原点）")
        
        # 获取有效的x坐标
        while True:
            try:
                start_x = int(input(f"请输入x坐标 (0-{w_dith-200}): "))
                if 0 <= start_x <= w_dith-200:
                    break
                print(f"x坐标应在0到{w_dith-200}之间")
            except ValueError:
                print("请输入有效的整数")
        
        # 获取有效的y坐标（转换为从上到下的坐标系）
        while True:
            try:
                # 用户输入的是以左下角为原点的y值
                user_y = int(input(f"请输入y坐标 (0-{h_dith-200}): "))
                if 0 <= user_y <= h_dith-200:
                    # 转换为图像坐标系（左上角为原点）
                    start_y = h_dith - 200 - user_y
                    break
                print(f"y坐标应在0到{h_dith-200}之间")
            except ValueError:
                print("请输入有效的整数")
        
        print(f"将从坐标({start_x}, {user_y})开始裁剪200x200区域")

    # 裁剪处理后的图像
    cropped_image = dithered_image[start_y:start_y+200, start_x:start_x+200]

# 保存裁剪后的图像
cv2.imwrite(os.path.join(output_dir, 'final_image_200x200.png'), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))

# 显示裁剪后的图像
plt.figure(figsize=(8, 8))
plt.imshow(cropped_image)
plt.title('最终裁剪后的图像 (200x200)')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'final_crop.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n所有结果已保存到: {output_dir}")
print(f"最终200x200的图像已保存为: final_image_200x200.png")
print("抖动算法在视觉上能提供平滑的过渡效果，尤其是在颜色渐变区域。")

# --- 5. 将裁剪后的图像转换为墨水屏数据包 ---
print("\n开始生成墨水屏数据包...")

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

# 计算需要的总字节数
total_bytes = (200 * 200 * 2) // 8  # 每个像素2位，8位一个字节
print(f"图像数据总字节数: {total_bytes} 字节")

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

# 创建最后一个数据包 (0x32)
last_packet_number = 0x32
last_packet_data = [ord(c) for c in "screen updating\r\n"]  # 转换字符串为ASCII码
last_packet_checksum = (last_packet_number + sum(last_packet_data)) % 256
last_packet = [last_packet_number] + last_packet_data + [last_packet_checksum]
packets.append(last_packet)

# 保存数据包到二进制文件
packets_file_path = os.path.join(output_dir, 'eink_display_data.bin')
with open(packets_file_path, 'wb') as f:
    for packet in packets:
        f.write(bytes(packet))

print(f"已生成 {len(packets)} 个数据包，共 {len(packets) * 202} 字节")
print(f"数据包已保存到: {packets_file_path}")

# 保存数据包的十六进制表示到文本文件，方便查看
hex_file_path = os.path.join(output_dir, 'eink_display_data_hex.txt')
with open(hex_file_path, 'w') as f:
    for i, packet in enumerate(packets):
        f.write(f"数据包 {i} (0x{packet[0]:02X}):\n")
        hex_data = ' '.join([f"{byte:02X}" for byte in packet])
        f.write(hex_data + '\n\n')

print(f"数据包的十六进制表示已保存到: {hex_file_path}")

# 显示数据包信息
print("\n数据包结构:")
print("每个数据包 202 字节: 1字节序号 + 200字节图像数据 + 1字节校验和")
print("前50个数据包 (0x00-0x31): 包含图像数据")
print("最后一个数据包 (0x32): 包含 'screen updating\r\n' 命令")
print(f"总共 {len(packets)} 个数据包，{len(packets) * 202} 字节")

# --- 6. 处理HEX数据文件并生成Python列表 ---

# 读取HEX数据文件
def process_hex_data(file_path):
    hex_packets = []
    current_packet = None
    
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

# 处理数据并保存为Python列表格式
def save_as_python_list(hex_packets, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# 墨水屏数据包列表 (不包含最后的固定数据包 0x32)\n\n')
        f.write('eink_display_data = [\n')
        
        for i, packet in enumerate(hex_packets):
            f.write(f'    "{packet}"')
            if i < len(hex_packets) - 1:
                f.write(',')
            f.write('\n')
            
        f.write(']\n')

# 处理数据并保存为纯文本格式
def save_as_txt(hex_packets, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('# 墨水屏数据包列表 (不包含最后的固定数据包 0x32)\n\n')
        f.write('eink_display_data = [\n')
        
        for i, packet in enumerate(hex_packets):
            # 每个数据包作为字符串元素，使用逗号分隔，最后一个元素不加逗号
            if i < len(hex_packets) - 1:
                f.write(f'    "{packet}",\n')
            else:
                f.write(f'    "{packet}"\n')
        
        f.write(']\n')
            
    print(f"已将数据包保存为Python列表格式: {output_path}")
    print("数据包以字符串形式存储在列表中，可直接在Python中导入使用。")

# 处理HEX数据文件并生成Python列表和TXT文件
print("\n开始处理HEX数据文件...")
eink_list_file = os.path.join(output_dir, 'eink_display_data_list.py')
eink_txt_file = os.path.join(output_dir, 'eink_display_data_list.txt')
hex_packets = process_hex_data(hex_file_path)
print(f"提取了 {len(hex_packets)} 个数据包")

# 保存为Python列表格式
save_as_python_list(hex_packets, eink_list_file)
print(f"已将处理结果保存到Python文件: {eink_list_file}")

# 保存为纯文本格式
save_as_txt(hex_packets, eink_txt_file)

print("\n处理完成！所有功能已整合到一个文件中。")