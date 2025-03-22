# 通过使用 K-means 聚类算法对图像进行颜色聚类分析，并生成一个基于聚类中心(即最具代表性的颜色)的RGB值和调色板。
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

# 查找photo.*文件
# photo_files = glob.glob(os.path.join(current_dir, 'photo.*'))
photo_files = glob.glob(os.path.join(current_dir, 'photo2.*')) # 调试用

if not photo_files:
    raise FileNotFoundError('未在当前目录找到photo图片文件，请确保存在photo.jpg或photo.png等格式的图片')

# 使用找到的第一个photo文件
photo_path = photo_files[0]

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

# K-means聚类，n_init设置为auto
n_colors = 5  # 提取的颜色数量
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto').fit(pixels)

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
plt.savefig('/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/color_mapping.png', dpi=300, bbox_inches='tight')
plt.show()

# 将原图像的每个像素映射到最接近的目标颜色，生成适合开发板显示的图像
labels = kmeans.labels_
mapped_image = np.zeros_like(image)

for i in range(h):
    for j in range(w):
        pixel_cluster = labels[i * w + j]
        mapped_image[i, j] = mapped_palette[pixel_cluster]

# 显示原图和映射后的图像对比
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mapped_image)
plt.title('映射后的图像（仅使用黑、白、红、黄四色）')
plt.axis('off')

plt.tight_layout()
plt.savefig('/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/mapped_image.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存映射后的图像，可以用于开发板
cv2.imwrite('/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/mapped_image_for_device.png',
            cv2.cvtColor(mapped_image, cv2.COLOR_RGB2BGR))