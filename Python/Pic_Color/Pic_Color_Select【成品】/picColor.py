# 通过使用 K-means 聚类算法对图像进行颜色聚类分析，并生成一个基于聚类中心(即最具代表性的颜色)的RGB值和调色板。
# 使用抖动算法对颜色进行处理
# 导入必要的库
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

print(f"图片尺寸: {w}x{h}像素")

# 重塑图像为二维数组，每行代表一个像素的RGB值
pixels = image.reshape(h * w, d)

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

# 显示原始图像、原始调色板和映射后的调色板
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(3, 1, 2)
plt.imshow(palette_image)
plt.title('提取的原始颜色调色板')
plt.axis('off')

plt.subplot(3, 1, 3)
plt.imshow(mapped_palette_image)
plt.title('映射到开发板支持颜色的调色板')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'color_palette.png'), dpi=300, bbox_inches='tight')
plt.show()

# 抖动算法处理
print("\n开始进行抖动算法处理...")

# 将目标颜色转换为数组形式，便于计算
target_colors_array = np.array(list(target_colors.values()))

# 创建一个新的图像用于抖动处理
dithered_image = image.copy().astype(np.float32)

# 图像预处理 - 使用双边滤波进行边缘保持平滑
print("正在进行图像预处理...")
dithered_image = cv2.bilateralFilter(dithered_image, d=5, sigmaColor=75, sigmaSpace=75)
print("图像预处理完成！")

# 定义Jarvis-Judice-Ninke抖动矩阵权重
jjn_weights = [
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
]
jjn_weights = np.array(jjn_weights) / 48  # 归一化权重

# 应用改进的抖动算法
total_pixels = h * w
processed_pixels = 0
print_interval = total_pixels // 100  # 每处理1%的像素打印一次进度

print("\n开始应用抖动算法...")
for y in range(h):
    for x in range(w):
        # 获取当前像素
        old_pixel = dithered_image[y, x].copy()
        
        # 更新和显示进度
        processed_pixels += 1
        if processed_pixels % print_interval == 0:
            progress = (processed_pixels / total_pixels) * 100
            print(f"处理进度: {progress:.1f}%")

        # 考虑局部区域的颜色分布
        local_region = dithered_image[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
        local_mean = np.mean(local_region, axis=(0, 1))
        old_pixel = old_pixel * 0.7 + local_mean * 0.3  # 权重混合

        # 找到最接近的目标颜色
        distances = np.sqrt(np.sum((old_pixel - target_colors_array) ** 2, axis=1))
        closest_idx = np.argmin(distances)
        new_pixel = target_colors_array[closest_idx]

        # 更新当前像素
        dithered_image[y, x] = new_pixel

        # 计算量化误差
        quant_error = old_pixel - new_pixel

        # 使用Jarvis-Judice-Ninke抖动模式分散误差
        for dy in range(3):
            for dx in range(5):
                if jjn_weights[dy, dx] > 0:  # 只处理有权重的位置
                    ny, nx = y + dy, x + dx - 2  # -2是为了使矩阵中心对齐当前像素
                    if 0 <= ny < h and 0 <= nx < w:
                        dithered_image[ny, nx] += quant_error * jjn_weights[dy, dx]

# 将浮点数转换回uint8
dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)

# 显示原图和处理后的图像
plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(dithered_image)
plt.title('抖动处理后（四色）')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'processed_image.png'), dpi=300, bbox_inches='tight')
plt.show()

# 裁剪图像到200x200像素
print("\n需要将处理后的图像裁剪为200x200像素")
print(f"原图尺寸: {w}x{h}像素")

# 计算图片中心位置
center_x, center_y = w // 2, h // 2

# 询问用户是否使用默认中心裁剪
use_center = input("默认将以图片中心点进行200x200像素的裁剪，是否同意？(y/n): ").lower().strip()

if use_center == 'y' or use_center == '':
    # 以图片中心点为中心进行裁剪
    start_x = max(0, center_x - 100)
    start_y = max(0, center_y - 100)
    
    # 调整以确保能裁剪到完整的200x200区域
    if start_x + 200 > w:
        start_x = w - 200
    if start_y + 200 > h:
        start_y = h - 200
        
    print(f"将以中心点({center_x}, {center_y})为中心进行裁剪")
else:
    # 以左下角为原点进行裁剪
    print(f"图片尺寸为{w}x{h}，请指定裁剪起始点坐标（左下角为原点）")
    
    # 获取有效的x坐标
    while True:
        try:
            start_x = int(input(f"请输入x坐标 (0-{w-200}): "))
            if 0 <= start_x <= w-200:
                break
            print(f"x坐标应在0到{w-200}之间")
        except ValueError:
            print("请输入有效的整数")
    
    # 获取有效的y坐标（转换为从上到下的坐标系）
    while True:
        try:
            # 用户输入的是以左下角为原点的y值
            user_y = int(input(f"请输入y坐标 (0-{h-200}): "))
            if 0 <= user_y <= h-200:
                # 转换为图像坐标系（左上角为原点）
                start_y = h - 200 - user_y
                break
            print(f"y坐标应在0到{h-200}之间")
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