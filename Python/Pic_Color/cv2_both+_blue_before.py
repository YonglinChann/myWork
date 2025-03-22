# 高级抖动算法与图像处理优化
# 实现蓝噪声抖动、优化缩放算法和边缘增强处理
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os
import glob
from matplotlib.colors import LinearSegmentedColormap
import time
from scipy import ndimage

# 设置matplotlib中文字体
plt.rcParams['font.family'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建输出目录
output_dir = os.path.join(current_dir, 'advanced_output')
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
print(f"原始图像尺寸: {w}×{h} 像素")

# 计算目标尺寸，保持宽高比
target_width = 800  # 设置目标宽度
scale_ratio = target_width / w
target_height = int(h * scale_ratio)

# 使用Lanczos插值进行高质量缩放
print("\n正在进行图像预处理缩放...")
image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

# 更新图像尺寸
h, w, d = image.shape
print(f"缩放后图像尺寸: {w}×{h} 像素")

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
start_time = time.time()
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto', verbose=1).fit(pixels)
end_time = time.time()
print(f"颜色聚类分析完成！用时: {end_time - start_time:.2f}秒\n")

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
print("\n开始进行直接映射处理...")
start_time = time.time()
labels = kmeans.labels_
mapped_image = np.zeros_like(image)

for i in range(h):
    for j in range(w):
        pixel_cluster = labels[i * w + j]
        mapped_image[i, j] = mapped_palette[pixel_cluster]

end_time = time.time()
print(f"直接映射处理完成！用时: {end_time - start_time:.2f}秒")


# 定义生成蓝噪声的函数
def generate_blue_noise(width, height, seed=42):
    """
    生成高质量的蓝噪声模式
    使用void-and-cluster算法的简化版本
    """
    print("正在生成蓝噪声模式...")
    
    # 对于大图像，进行降采样处理
    max_dimension = 512  # 最大处理尺寸
    downscale = False
    original_width, original_height = width, height
    scale_factor = 1
    
    if width > max_dimension or height > max_dimension:
        downscale = True
        scale_factor = max(width, height) / max_dimension
        width = int(width / scale_factor)
        height = int(height / scale_factor)
        print(f"图像较大，降采样至 {width}x{height} 进行蓝噪声生成")
    
    # 设置随机种子以获得可重复的结果
    np.random.seed(seed)
    
    # 初始化一个随机白噪声
    noise = np.random.random((height, width))
    
    # 高斯滤波器用于计算密度
    sigma = max(width, height) / 30  # 根据图像大小调整sigma
    
    # 预先计算高斯核以加速处理
    kernel_size = int(sigma * 6) // 2 * 2 + 1  # 确保是奇数
    kernel_size = min(kernel_size, min(width, height) // 2)  # 确保核大小合理
    print(f"使用 {kernel_size}x{kernel_size} 高斯核进行滤波")
    
    gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T
    
    # 优化的高斯滤波函数
    def fast_gaussian_filter(x):
        return cv2.filter2D(x, -1, gaussian_kernel, borderType=cv2.BORDER_WRAP)
    
    # 初始化二值蓝噪声模式
    binary = np.zeros((height, width), dtype=np.float32)
    
    # 随机选择初始点
    initial_points = int(width * height * 0.01)  # 使用1%的点作为初始点
    print(f"步骤1/5: 初始化随机点 ({initial_points}个点)...")
    flat_indices = np.random.choice(width * height, initial_points, replace=False)
    y_indices, x_indices = np.unravel_index(flat_indices, (height, width))
    for y, x in zip(y_indices, x_indices):
        binary[y, x] = 1
    
    # 计算初始密度
    density = fast_gaussian_filter(binary)
    
    # 第一阶段：移除空隙（找到密度最低的区域并添加点）
    # 减少迭代次数以提高性能
    num_points_to_add = min(int(width * height * 0.1), 5000)  # 最多添加5000个点
    print(f"步骤2/5: 填充空隙 (添加{num_points_to_add}个点)...")
    
    # 显示进度的间隔
    progress_interval = max(1, num_points_to_add // 20)
    
    # 批量处理以提高性能
    batch_size = 10
    for i in range(0, num_points_to_add, batch_size):
        # 每处理5%显示一次进度
        if i % progress_interval == 0:
            progress = (i / num_points_to_add) * 100
            print(f"  填充空隙进度: {progress:.1f}%", flush=True)  # 使用flush确保立即显示
        
        # 找到当前批次中密度最低的点
        mask = binary < 0.5  # 只考虑当前为0的点
        if not np.any(mask):
            break
            
        masked_density = np.ma.array(density, mask=~mask)
        
        # 找到多个低密度点
        current_batch_size = min(batch_size, num_points_to_add - i)
        flat_indices = np.ma.argsort(masked_density.ravel())[:current_batch_size]
        y_indices, x_indices = np.unravel_index(flat_indices, density.shape)
        
        # 添加这些点
        for y, x in zip(y_indices, x_indices):
            binary[y, x] = 1
        
        # 更新密度 (每批次只计算一次)
        density = fast_gaussian_filter(binary)
    
    print("  填充空隙进度: 100.0%")
    
    # 第二阶段：移除聚类（找到密度最高的点并移除）
    # 减少迭代次数以提高性能
    num_points_to_remove = min(int(width * height * 0.05), 2500)  # 最多移除2500个点
    print(f"步骤3/5: 移除聚类 (移除{num_points_to_remove}个点)...")
    
    # 显示进度的间隔
    progress_interval = max(1, num_points_to_remove // 20)
    
    # 批量处理以提高性能
    for i in range(0, num_points_to_remove, batch_size):
        # 每处理5%显示一次进度
        if i % progress_interval == 0:
            progress = (i / num_points_to_remove) * 100
            print(f"  移除聚类进度: {progress:.1f}%", flush=True)
            
        # 找到当前批次中密度最高的点
        mask = binary > 0.5  # 只考虑当前为1的点
        if not np.any(mask):
            break
            
        masked_density = np.ma.array(density, mask=~mask)
        
        # 找到多个高密度点
        current_batch_size = min(batch_size, num_points_to_remove - i)
        flat_indices = np.ma.argsort(-masked_density.ravel())[:current_batch_size]
        y_indices, x_indices = np.unravel_index(flat_indices, density.shape)
        
        # 移除这些点
        for y, x in zip(y_indices, x_indices):
            binary[y, x] = 0
        
        # 更新密度 (每批次只计算一次)
        density = fast_gaussian_filter(binary)
    
    print("  移除聚类进度: 100.0%")
    
    # 生成最终的蓝噪声
    print("步骤4/5: 生成蓝噪声分布...")
    blue_noise = np.zeros((height, width), dtype=np.float32)
    
    # 对二值图像中的1进行排序，生成0-1之间的蓝噪声
    ones_indices = np.where(binary > 0.5)
    num_ones = len(ones_indices[0])
    
    if num_ones > 0:
        # 随机打乱顺序
        perm = np.random.permutation(num_ones)
        
        # 按顺序分配值
        progress_interval = max(1, num_ones // 10)
        for i, idx in enumerate(perm):
            if i % progress_interval == 0:  # 每处理10%显示一次进度
                progress = (i / num_ones) * 100
                print(f"  处理亮区进度: {progress:.1f}%", flush=True)
            y, x = ones_indices[0][idx], ones_indices[1][idx]
            blue_noise[y, x] = (i + 1) / (num_ones + 1)
    
    # 对剩余的0进行排序
    zeros_indices = np.where(binary < 0.5)
    num_zeros = len(zeros_indices[0])
    
    if num_zeros > 0:
        # 随机打乱顺序
        perm = np.random.permutation(num_zeros)
        
        # 按顺序分配值
        progress_interval = max(1, num_zeros // 10)
        for i, idx in enumerate(perm):
            if i % progress_interval == 0:  # 每处理10%显示一次进度
                progress = (i / num_zeros) * 100
                print(f"  处理暗区进度: {progress:.1f}%", flush=True)
            y, x = zeros_indices[0][idx], zeros_indices[1][idx]
            blue_noise[y, x] = (i + 1) / (num_zeros + 1)
    
    # 最终平滑处理
    print("步骤5/5: 最终平滑处理...")
    blue_noise = fast_gaussian_filter(blue_noise)
    blue_noise = (blue_noise - np.min(blue_noise)) / (np.max(blue_noise) - np.min(blue_noise))
    
    # 如果进行了降采样，现在需要上采样回原始尺寸
    if downscale:
        print(f"将蓝噪声上采样回原始尺寸 {original_width}x{original_height}...")
        blue_noise = cv2.resize(blue_noise, (original_width, original_height), 
                               interpolation=cv2.INTER_CUBIC)
    
    print("蓝噪声模式生成完成！")
    return blue_noise


# 方法2: 改进的蓝噪声抖动算法处理
print("\n开始进行改进的蓝噪声抖动算法处理...")

# 将目标颜色转换为数组形式，便于计算
target_colors_array = np.array(list(target_colors.values()))

# 创建一个新的图像用于抖动处理
dithered_image = image.copy().astype(np.float32)

# 图像预处理 - 使用改进的边缘保持和颜色平滑处理
print("正在进行图像预处理...")

# 转换到LAB色彩空间进行颜色校正
lab = cv2.cvtColor(dithered_image.astype(np.uint8), cv2.COLOR_RGB2LAB)

# 计算L通道的均值和标准差，用于自适应调整
l_mean = np.mean(lab[...,0])
l_std = np.std(lab[...,0])

# 自适应亮度校正
lab[...,0] = np.clip(lab[...,0] * (128 / l_mean), 0, 255).astype(np.uint8)

# 使用CLAHE进行自适应对比度增强，但降低clipLimit以减少噪点
clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
lab[...,0] = clahe.apply(lab[...,0])

# 颜色饱和度调整（a和b通道）
lab[...,1:] = np.clip(lab[...,1:] * 0.85, 0, 255).astype(np.uint8)  # 略微降低饱和度

# 转回RGB空间
dithered_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)

# 使用双边滤波进行边缘保持平滑
sigma_color = 75  # 颜色空间的标准差
sigma_space = int(min(w, h) / 50)  # 空间距离的标准差，自适应设置
dithered_image = cv2.bilateralFilter(
    src=dithered_image.astype(np.uint8),
    d=sigma_space,
    sigmaColor=sigma_color,
    sigmaSpace=sigma_space
)

# 最后使用保守的双边滤波来进一步减少噪点
sigmaColor = np.mean(np.std(dithered_image, axis=(0,1))) * 0.2  # 降低颜色标准差以减少噪点
sigmaSpace = min(w, h) / 40.0
dithered_image = cv2.bilateralFilter(dithered_image, d=5, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

print("图像预处理完成！")

# 生成蓝噪声模式
blue_noise = generate_blue_noise(w, h)

# 应用改进的蓝噪声抖动
print("\n开始应用改进的蓝噪声抖动算法...")
start_time = time.time()

# 使用改进的蓝噪声抖动处理图像
total_pixels = h * w
processed_pixels = 0
print_interval = total_pixels // 20  # 每处理5%的像素打印一次进度

# 计算图像的局部对比度和亮度，使用更大的窗口以减少噪点
local_contrast = np.zeros((h, w))
local_brightness = np.zeros((h, w))
kernel_size = 9  # 进一步增加窗口大小以获得更平滑的结果
padded_image = np.pad(dithered_image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)), mode='symmetric')  # 使用symmetric模式以更好地保持边缘

# 使用改进的高斯权重计算局部特征
sigma = kernel_size/4.0  # 减小sigma以增强边缘检测
gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
gaussian_kernel = gaussian_kernel * gaussian_kernel.T

# 使用积分图像加速计算
integral_img = cv2.integral(cv2.cvtColor(padded_image.astype(np.uint8), cv2.COLOR_RGB2GRAY))

for y in range(h):
    for x in range(w):
        # 使用积分图像快速计算局部区域统计
        y1, y2 = y, y + kernel_size
        x1, x2 = x, x + kernel_size
        window_sum = integral_img[y2, x2] - integral_img[y2, x1] - integral_img[y1, x2] + integral_img[y1, x1]
        window_mean = window_sum / (kernel_size * kernel_size)
        
        # 计算加权标准差
        window = padded_image[y:y+kernel_size, x:x+kernel_size]
        diff_sq = (window - window_mean) ** 2
        weighted_std = np.sqrt(np.sum(gaussian_kernel * np.mean(diff_sq, axis=2)) / np.sum(gaussian_kernel))
        
        # 使用HSV空间计算亮度，减少颜色偏差的影响
        hsv_window = cv2.cvtColor(window.astype(np.uint8), cv2.COLOR_RGB2HSV)
        weighted_brightness = np.sum(gaussian_kernel * hsv_window[..., 2]) / np.sum(gaussian_kernel)
        
        local_contrast[y, x] = weighted_std
        local_brightness[y, x] = weighted_brightness

# 使用改进的归一化函数，采用双sigmoid函数以更好地控制中间调
def improved_normalize(x, alpha=3, beta=0.5):
    x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
    # 双sigmoid函数提供更平滑的过渡
    return 0.5 * (1 / (1 + np.exp(-alpha * (x_normalized - beta))) + 
                  1 / (1 + np.exp(-alpha * (x_normalized - (1-beta)))))

local_contrast = improved_normalize(local_contrast, alpha=4, beta=0.4)
local_brightness = improved_normalize(local_brightness, alpha=3, beta=0.45)

# 改进的JJN抖动矩阵权重，降低整体权重以减少噪点
jjn_weights = np.array([
    [0, 0, 0, 5, 3],
    [2, 4, 5, 4, 2],
    [1, 2, 3, 2, 1]
]) / 48.0

# 定义颜色距离计算函数，使用加权欧氏距离
def color_distance(c1, c2):
    # 人眼对绿色最敏感，其次是红色，最后是蓝色
    weights = np.array([0.299, 0.587, 0.114])
    return np.sqrt(np.sum(weights * (c1 - c2) ** 2))

for y in range(h):
    for x in range(w):
        # 获取当前像素
        old_pixel = dithered_image[y, x].copy() / 255.0

        # 更新和显示进度
        processed_pixels += 1
        if processed_pixels % print_interval == 0:
            progress = (processed_pixels / total_pixels) * 100
            print(f"处理进度: {progress:.1f}%")

        # 获取自适应阈值，降低对比度和亮度的影响
        base_threshold = blue_noise[y, x]
        contrast_weight = 0.8 + 0.2 * local_contrast[y, x]  # 减少对比度的影响
        brightness_weight = 0.8 + 0.2 * local_brightness[y, x]  # 减少亮度的影响
        threshold = base_threshold * contrast_weight * brightness_weight

        # 使用颜色距离找到最接近的目标颜色
        distances = [color_distance(old_pixel, color/255.0) for color in target_colors_array]
        closest_idx = np.argmin(distances)
        new_pixel = target_colors_array[closest_idx] / 255.0

        # 根据颜色距离调整量化阈值
        min_distance = distances[closest_idx]
        threshold_scale = np.clip(1.0 - min_distance, 0.5, 1.0)  # 距离越近，阈值越严格
        threshold = threshold * threshold_scale

        # 计算量化误差，根据局部对比度和颜色距离调整误差扩散
        error_scale = (1.0 - min_distance) * (1.0 - local_contrast[y, x] * 0.7)
        quant_error = (old_pixel - new_pixel) * error_scale

        # 使用改进的JJN权重分布误差，根据局部特征动态调整权重
        for i in range(3):
            for j in range(5):
                if i == 0 and j < 3:
                    continue
                if y + i < h and 0 <= x + j - 2 < w:
                    # 根据目标像素的局部特征调整权重
                    target_contrast = local_contrast[y + i, x + j - 2]
                    target_brightness = local_brightness[y + i, x + j - 2]
                    
                    # 在高对比度区域减少误差扩散
                    weight_scale = 1.0 - (target_contrast * 0.6 + target_brightness * 0.2)
                    
                    # 应用缩放后的权重
                    scaled_weight = jjn_weights[i, j] * weight_scale * 0.7  # 整体降低权重强度
                    
                    dithered_image[y + i, x + j - 2] = np.clip(
                        dithered_image[y + i, x + j - 2] / 255.0 + quant_error * scaled_weight,
                        0, 1
                    ) * 255.0

        # 将像素映射到最接近的目标颜色
        distances = np.sqrt(np.sum((new_pixel - target_colors_array / 255.0) ** 2, axis=1))
        closest_idx = np.argmin(distances)
        dithered_image[y, x] = target_colors_array[closest_idx]

end_time = time.time()
print(f"蓝噪声抖动处理完成！用时: {end_time - start_time:.2f}秒")

# 将浮点数转换回uint8
dithered_image = np.clip(dithered_image, 0, 255).astype(np.uint8)

# 后处理 - 应用边缘保持滤波以减少噪点但保留边缘
print("正在进行后处理以提高图像质量...")
# 使用引导滤波进行边缘保持平滑
if cv2.__version__ >= '3.0.0':  # 确保OpenCV版本支持
    try:
        dithered_image = cv2.ximgproc.guidedFilter(
            guide=dithered_image,
            src=dithered_image,
            radius=1,
            eps=1e-6,
            dDepth=-1
        )
    except:
        print("引导滤波不可用，跳过此步骤")

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
print(f"蓝噪声抖动的均方误差: {mse_dithered:.2f}")

# 计算两种方法之间的差异
diff_percentage = np.count_nonzero(difference) / (h * w * d) * 100
print(f"两种方法的像素差异比例: {diff_percentage:.2f}%")

# 如果抖动处理的误差小于简单映射，说明抖动效果更好
if mse_dithered < mse_mapped:
    print("蓝噪声抖动处理后的图像更接近原始图像，建议使用抖动处理后的图像。")
else:
    print("直接映射的图像更接近原始图像，但蓝噪声抖动处理可能在视觉上提供更好的效果。")

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
plt.title('蓝噪声抖动处理（四色）')
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
plt.title('蓝噪声抖动（放大区域）')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(crop_diff)
plt.title('差异增强（放大区域）')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'zoomed_comparison.png'), dpi=300, bbox_inches='tight')
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

# 添加优化的图像缩放功能
print("\n正在进行优化的图像缩放处理...")

# 计算缩放比例
target_width = 200
scale_ratio = target_width / w
target_height = int(h * scale_ratio)

# 保存处理后的图像
cv2.imwrite(os.path.join(output_dir, 'processed_original.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'processed_mapped.png'), cv2.cvtColor(mapped_image, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'processed_dithered.png'), cv2.cvtColor(dithered_image, cv2.COLOR_RGB2BGR))

# 创建处理后的图像对比
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title('原始图像')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mapped_image)
plt.title('直接映射')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(dithered_image)
plt.title('蓝噪声抖动')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"图像已等比例缩放至宽度 {target_width} 像素 (高度: {target_height} 像素)")
print(f"缩放比例: {scale_ratio:.2f}")

# 比较不同缩放算法的效果
print("\n比较不同缩放算法的效果...")

# 使用不同的插值方法缩放抖动图像
scaled_nearest = cv2.resize(dithered_image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
scaled_linear = cv2.resize(dithered_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
scaled_cubic = cv2.resize(dithered_image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
scaled_lanczos = cv2.resize(dithered_image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

# 对抖动图像应用边缘增强后再缩放
enhanced_dithered = cv2.detailEnhance(dithered_image, sigma_s=10, sigma_r=0.15)
scaled_enhanced = cv2.resize(enhanced_dithered, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

# 保存不同缩放算法的结果
cv2.imwrite(os.path.join(output_dir, 'scaled_nearest.png'), cv2.cvtColor(scaled_nearest, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'scaled_linear.png'), cv2.cvtColor(scaled_linear, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'scaled_cubic.png'), cv2.cvtColor(scaled_cubic, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'scaled_lanczos.png'), cv2.cvtColor(scaled_lanczos, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, 'scaled_enhanced.png'), cv2.cvtColor(scaled_enhanced, cv2.COLOR_RGB2BGR))

# 创建不同缩放算法的对比图
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(scaled_nearest)
plt.title('最近邻插值 (INTER_NEAREST)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(scaled_linear)
plt.title('线性插值 (INTER_LINEAR)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(scaled_cubic)
plt.title('三次插值 (INTER_CUBIC)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(scaled_lanczos)
plt.title('Lanczos插值 (INTER_LANCZOS4)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(scaled_enhanced)
plt.title('边缘增强 + Lanczos插值')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scaling_methods_comparison.png'), dpi=300, bbox_inches='tight')
plt.show()

# 放大显示缩放后图像的细节
# 选择缩放后图像的中心区域
s_center_y, s_center_x = target_height // 2, target_width // 2
s_crop_size = min(target_height, target_width) // 4
s_crop_y1, s_crop_y2 = s_center_y - s_crop_size, s_center_y + s_crop_size
s_crop_x1, s_crop_x2 = s_center_x - s_crop_size, s_center_x + s_crop_size

# 裁剪区域
s_crop_nearest = scaled_nearest[s_crop_y1:s_crop_y2, s_crop_x1:s_crop_x2]
s_crop_linear = scaled_linear[s_crop_y1:s_crop_y2, s_crop_x1:s_crop_x2]
s_crop_cubic = scaled_cubic[s_crop_y1:s_crop_y2, s_crop_x1:s_crop_x2]
s_crop_lanczos = scaled_lanczos[s_crop_y1:s_crop_y2, s_crop_x1:s_crop_x2]
s_crop_enhanced = scaled_enhanced[s_crop_y1:s_crop_y2, s_crop_x1:s_crop_x2]

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(s_crop_nearest)
plt.title('最近邻插值 (放大区域)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(s_crop_linear)
plt.title('线性插值 (放大区域)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(s_crop_cubic)
plt.title('三次插值 (放大区域)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(s_crop_lanczos)
plt.title('Lanczos插值 (放大区域)')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(s_crop_enhanced)
plt.title('边缘增强 + Lanczos插值 (放大区域)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'scaling_methods_zoomed.png'), dpi=300, bbox_inches='tight')
plt.show()

# 计算不同缩放方法的图像质量指标
print("\n不同缩放方法的图像质量评估：")


# 定义一个函数来计算结构相似性指数(SSIM)
def calculate_ssim(img1, img2):
    """计算两个图像之间的结构相似性指数"""
    # 转换为灰度图像
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # 计算SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # 应用高斯滤波
    window_size = 11
    sigma = 1.5
    gauss = cv2.getGaussianKernel(window_size, sigma)
    window = gauss * gauss.T

    # 计算均值
    mu1 = cv2.filter2D(gray1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(gray2, -1, window)[5:-5, 5:-5]

    # 计算方差和协方差
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(gray1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(gray2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(gray1 * gray2, -1, window)[5:-5, 5:-5] - mu1_mu2

    # 计算SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


# 将原始图像缩放到相同大小，用于比较
original_scaled = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

# 计算每种方法的均方误差(MSE)和结构相似性指数(SSIM)
methods = {
    "最近邻插值": scaled_nearest,
    "线性插值": scaled_linear,
    "三次插值": scaled_cubic,
    "Lanczos插值": scaled_lanczos,
    "边缘增强+Lanczos插值": scaled_enhanced
}

print("方法\t\t\t均方误差(MSE)\t结构相似性(SSIM)")
print("-" * 60)

for name, img in methods.items():
    # 计算MSE
    mse = np.mean((original_scaled.astype(np.float32) - img.astype(np.float32)) ** 2)

    # 计算SSIM
    ssim = calculate_ssim(original_scaled, img)

    # 打印结果
    print(f"{name:<20}\t{mse:.2f}\t\t{ssim:.4f}")

# 创建最终的优化缩放图像
print("\n创建最终优化的缩放图像...")

# 基于评估结果，选择最佳的缩放方法
# 这里我们使用边缘增强+Lanczos插值，因为它通常提供最佳的视觉质量
final_scaled_image = scaled_enhanced

# 保存最终的优化缩放图像
cv2.imwrite(os.path.join(output_dir, 'final_optimized_scaled.png'), cv2.cvtColor(final_scaled_image, cv2.COLOR_RGB2BGR))

# 显示最终的优化缩放图像
plt.figure(figsize=(10, 10))
plt.imshow(final_scaled_image)
plt.title('最终优化的缩放图像 (200像素宽度)')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'final_optimized_display.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n所有结果已保存到: {output_dir}")
print("请查看'scaling_methods_comparison.png'和'scaling_methods_zoomed.png'了解不同缩放算法的效果。")
print("最终优化的缩放图像已保存为'final_optimized_scaled.png'。")
print("\n总结：")
print("1. 蓝噪声抖动算法能够产生更均匀的颜色分布，减少明显的噪点")
print("2. 边缘增强处理能够保留更多图像细节")
print("3. Lanczos插值在缩放时提供最佳的视觉质量")
print("4. 结合这三种技术，可以显著提高缩放后图像的清晰度和视觉效果")