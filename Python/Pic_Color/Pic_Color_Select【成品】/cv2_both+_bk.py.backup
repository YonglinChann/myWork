# 通过使用 K-means 聚类算法对图像进行颜色聚类分析，并生成一个基于聚类中心(即最具代表性的颜色)的RGB值和调色板。
# 同时比较直接映射和抖动算法两种方法的效果
# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os
import glob
from matplotlib.colors import LinearSegmentedColormap

# 设置matplotlib中文字体
plt.rcParams['font.family'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建输出目录
output_dir = os.path.join(current_dir, 'comparison_output')
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

# 显示可用的图片列表
print("可用的图片文件:\n")
for num in sorted(photo_dict.keys()):
    print(f"{num}: {os.path.basename(photo_dict[num])}")

# 询问用户要分析哪张图片
while True:
    try:
        photo_num = int(input("\n请输入要分析的图片编号: "))
        if photo_num in photo_dict:
            photo_path = photo_dict[photo_num]
            break
        else:
            print(f"错误: 未找到编号为 {photo_num} 的图片，请重新输入")
    except ValueError:
        print("错误: 请输入有效的整数")

print(f"将处理图片: {os.path.basename(photo_path)}")

# 使用OpenCV读取图像（BGR格式）
image = cv2.imread(photo_path)
if image is None:
    raise ValueError(f'无法读取图片文件：{photo_path}')

# 转换为RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像尺寸
h, w, d = image.shape
# 重塑图像为二维数组，每行代表一个像素的RGB值
pixels = image.reshape(h * w, d)

# 询问用户需要提取的颜色数量
while True:
    try:
        n_colors = int(input("\n请输入需要提取的颜色数量 (推荐3-10): "))
        if 2 <= n_colors <= 20:
            break
        print("颜色数量应在2-20之间")
    except ValueError:
        print("请输入有效的整数")

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

# 显示原始图像、原始调色板和映射后的调色板对比
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
plt.savefig(os.path.join(output_dir, 'color_mapping.png'), dpi=300, bbox_inches='tight')
plt.show()

# 方法1: 直接映射 - 将原图像的每个像素映射到最接近的目标颜色
labels = kmeans.labels_
mapped_image = np.zeros_like(image)

for i in range(h):
    for j in range(w):
        pixel_cluster = labels[i * w + j]
        mapped_image[i, j] = mapped_palette[pixel_cluster]

# 方法2: 改进的抖动算法处理
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

# 接第一部分代码，这部分主要进行对比分析和可视化

# 创建差异图像，突出显示两种方法的区别
difference = cv2.absdiff(mapped_image, dithered_image)
# 放大差异，使其更明显
difference_enhanced = cv2.convertScaleAbs(difference, alpha=5.0)

# 计算原始图像与处理后图像的差异
print("\n图像处理效果评估：")
# 计算简单映射与原图的均方误差
mse_mapped = np.mean((image.astype(np.float32) - mapped_image.astype(np.float32)) ** 2)
print(f"直接映射的均方误差: {mse_mapped:.2f}")

# 计算抖动处理与原图的均方误差
mse_dithered = np.mean((image.astype(np.float32) - dithered_image.astype(np.float32)) ** 2)
print(f"抖动处理的均方误差: {mse_dithered:.2f}")

# 计算两种方法之间的差异
diff_percentage = np.count_nonzero(difference) / (h * w * d) * 100
print(f"两种方法的像素差异比例: {diff_percentage:.2f}%")

# 如果抖动处理的误差小于简单映射，说明抖动效果更好
if mse_dithered < mse_mapped:
    print("抖动处理后的图像更接近原始图像，建议使用抖动处理后的图像。")
else:
    print("直接映射的图像更接近原始图像，但抖动处理可能在视觉上提供更好的效果。")

# 创建热力图，显示两种方法的差异区域
difference_gray = cv2.cvtColor(difference_enhanced, cv2.COLOR_RGB2GRAY)
# 创建自定义颜色映射，从透明到红色
colors = [(0, 0, 0, 0), (1, 0, 0, 1)]  # 从透明黑色到红色
cmap = LinearSegmentedColormap.from_list('diff_cmap', colors, N=256)

# 显示原图、直接映射和抖动处理后的图像对比
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(mapped_image)
plt.title('直接映射（四色）')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(dithered_image)
plt.title('抖动处理后（四色）')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(image)
plt.imshow(difference_gray, cmap=cmap, alpha=0.7)
plt.title('两种方法的差异区域（红色）')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 创建放大对比图，显示细节差异
# 选择图像中间区域进行放大
center_y, center_x = h // 2, w // 2
crop_size = min(h, w) // 4
crop_y1, crop_y2 = center_y - crop_size, center_y + crop_size
crop_x1, crop_x2 = center_x - crop_size, center_x + crop_size

# 裁剪区域
crop_original = image[crop_y1:crop_y2, crop_x1:crop_x2]
crop_mapped = mapped_image[crop_y1:crop_y2, crop_x1:crop_x2]
crop_dithered = dithered_image[crop_y1:crop_y2, crop_x1:crop_x2]
crop_diff = difference_enhanced[crop_y1:crop_y2, crop_x1:crop_x2]

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(crop_original)
plt.title('原始图像（放大区域）')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(crop_mapped)
plt.title('直接映射（放大区域）')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(crop_dithered)
plt.title('抖动处理（放大区域）')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(crop_diff)
plt.title('差异增强（放大区域）')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'zoomed_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 创建直方图比较
plt.figure(figsize=(15, 10))

# 原始图像的颜色分布
plt.subplot(2, 2, 1)
for i, color in enumerate(['r', 'g', 'b']):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title('原始图像颜色分布')
plt.xlim([0, 256])
plt.grid(True)

# 直接映射后的颜色分布
plt.subplot(2, 2, 2)
for i, color in enumerate(['r', 'g', 'b']):
    hist = cv2.calcHist([mapped_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title('直接映射后颜色分布')
plt.xlim([0, 256])
plt.grid(True)

# 抖动处理后的颜色分布
plt.subplot(2, 2, 3)
for i, color in enumerate(['r', 'g', 'b']):
    hist = cv2.calcHist([dithered_image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title('抖动处理后颜色分布')
plt.xlim([0, 256])
plt.grid(True)

# 差异图像的颜色分布
plt.subplot(2, 2, 4)
for i, color in enumerate(['r', 'g', 'b']):
    hist = cv2.calcHist([difference], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
plt.title('差异图像颜色分布')
plt.xlim([0, 256])
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'histogram_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 保存处理后的图像
print("\n正在保存处理结果...")
cv2.imwrite(os.path.join(output_dir, 'mapped_image.png'), cv2.cvtColor(mapped_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'dithered_image.png'), cv2.cvtColor(dithered_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'difference_enhanced.png'), cv2.cvtColor(difference_enhanced, cv2.COLOR_RGB2BGR))
print("图像保存完成！")

# 创建一个合并文件，将两种方法的结果并排显示
combined_image = np.hstack((mapped_image, dithered_image))
cv2.imwrite(os.path.join(output_dir, 'side_by_side_comparison.png'), cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

# 创建一个带标签的合并图像
h_combined = max(h, 50)  # 确保有足够的空间放置标签
w_combined = w * 2
combined_with_labels = np.ones((h_combined + 50, w_combined, 3), dtype=np.uint8) * 255

# 添加图像
combined_with_labels[:h, :w] = mapped_image
combined_with_labels[:h, w:] = dithered_image

# 添加标签
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(combined_with_labels, '直接映射', (w // 4, h + 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(combined_with_labels, '抖动处理', (w + w // 4, h + 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imwrite(os.path.join(output_dir, 'labeled_comparison.png'), cv2.cvtColor(combined_with_labels, cv2.COLOR_RGB2BGR))

# 添加图像缩放功能，将图像等比例缩放到宽度为200像素
print("\n正在进行图像缩放处理...")

# 计算缩放比例
target_width = 200
scale_ratio = target_width / w
target_height = int(h * scale_ratio)

# 缩放原始图像、直接映射图像和抖动处理图像
scaled_original = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
scaled_mapped = cv2.resize(mapped_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
scaled_dithered = cv2.resize(dithered_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

# 保存缩放后的图像
cv2.imwrite(os.path.join(output_dir, 'scaled_original.png'), cv2.cvtColor(scaled_original, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'scaled_mapped.png'), cv2.cvtColor(scaled_mapped, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'scaled_dithered.png'), cv2.cvtColor(scaled_dithered, cv2.COLOR_RGB2BGR))

# 创建缩放后的图像对比
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(scaled_original)
plt.title(f'原始图像 (缩放至 {target_width}x{target_height})')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(scaled_mapped)
plt.title(f'直接映射 (缩放至 {target_width}x{target_height})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(scaled_dithered)
plt.title(f'抖动处理 (缩放至 {target_width}x{target_height})')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scaling_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"图像已等比例缩放至宽度 {target_width} 像素 (高度: {target_height} 像素)")
print(f"缩放比例: {scale_ratio:.2f}")

print(f"\n所有结果已保存到: {output_dir}")
print("请查看对比图像，特别是放大区域和差异热力图，以便更清晰地看到两种方法的区别。")
print("抖动算法通常在视觉上能提供更平滑的过渡效果，尤其是在颜色渐变区域。")
print("缩放后的图像已保存，可以查看'scaling_comparison.png'了解缩放效果。")