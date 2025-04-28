# 验证墨水屏数据包的准确性
# 读取二进制数据包并将其转换回BMP图像

import os
import numpy as np
import matplotlib
# 设置matplotlib后端为Agg，避免显示图形界面
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import struct

# 设置matplotlib中文字体，防止中文乱码
plt.rcParams['font.family'] = ['Hiragino Sans GB']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']

# 墨水屏像素定义：
# 黑 00
# 白 01
# 黄 10
# 红 11

# 定义二进制值到RGB颜色的映射
binary_to_rgb = {
    '00': [0, 0, 0],       # 黑色
    '01': [255, 255, 255], # 白色
    '10': [255, 255, 0],   # 黄色
    '11': [255, 0, 0]      # 红色
}

# 使用固定路径而不是相对路径
current_dir = '/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/Pic_Color_Select【成品】'

# 输入和输出目录
input_dir = os.path.join(current_dir, 'output')
output_dir = os.path.join(current_dir, 'verification_output')
os.makedirs(output_dir, exist_ok=True)

# 二进制数据文件路径
bin_file_path = os.path.join(input_dir, 'eink_display_data.bin')

# 检查文件是否存在
if not os.path.exists(bin_file_path):
    raise FileNotFoundError(f'未找到二进制数据文件: {bin_file_path}')

print(f"正在读取二进制数据文件: {bin_file_path}")

# 读取二进制数据
with open(bin_file_path, 'rb') as f:
    binary_data = f.read()

# 解析数据包
packets = []
packet_size = 202  # 1字节序号 + 200字节数据 + 1字节校验和

for i in range(0, len(binary_data), packet_size):
    packet_data = binary_data[i:i+packet_size]
    if len(packet_data) == packet_size:
        packet_number = packet_data[0]
        payload = packet_data[1:-1]
        checksum = packet_data[-1]
        
        # 验证校验和
        calculated_checksum = (packet_number + sum(payload)) % 256
        checksum_valid = (calculated_checksum == checksum)
        
        packets.append({
            'number': packet_number,
            'payload': payload,
            'checksum': checksum,
            'checksum_valid': checksum_valid
        })

print(f"共解析出 {len(packets)} 个数据包")

# 检查最后一个数据包是否为命令包
last_packet = packets[-1]
if last_packet['number'] == 0x32:
    command_text = ''.join([chr(b) for b in last_packet['payload']])
    print(f"最后一个数据包是命令包: {command_text}")
    # 移除命令包，只处理图像数据包
    packets = packets[:-1]

# 验证所有数据包的校验和
all_checksums_valid = all(packet['checksum_valid'] for packet in packets)
print(f"所有数据包校验和验证: {'通过' if all_checksums_valid else '失败'}")

# 提取图像数据
image_data = bytearray()
for packet in packets:
    image_data.extend(packet['payload'])

# 将字节数据转换为二进制字符串
binary_string = ''
for byte in image_data:
    # 将每个字节转换为8位二进制字符串
    binary_string += format(byte, '08b')

# 创建200x200的图像数组
image_array = np.zeros((200, 200, 3), dtype=np.uint8)

# 填充图像数组
pixel_index = 0
for y in range(200):
    for x in range(200):
        if pixel_index + 2 <= len(binary_string):
            pixel_binary = binary_string[pixel_index:pixel_index+2]
            image_array[y, x] = binary_to_rgb[pixel_binary]
            pixel_index += 2

# 显示重建的图像
plt.figure(figsize=(8, 8))
plt.imshow(image_array)
plt.title('从二进制数据重建的图像')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reconstructed_image.png'), dpi=300, bbox_inches='tight')

# 保存为BMP文件
reconstructed_image = Image.fromarray(image_array)
reconstructed_image.save(os.path.join(output_dir, 'reconstructed_image.bmp'))

# 如果原始处理后的图像存在，加载它进行比较
original_image_path = os.path.join(input_dir, 'final_image_200x200.png')
if os.path.exists(original_image_path):
    # 加载原始图像
    from matplotlib.image import imread
    original_image = imread(original_image_path)
    
    # 显示原始图像和重建图像的比较
    plt.figure(figsize=(15, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('原始处理后的图像')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image_array)
    plt.title('从二进制数据重建的图像')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300, bbox_inches='tight')
    # 不使用plt.show()，因为使用了Agg后端
    
    # 计算差异
    if original_image.shape == image_array.shape:
        # 计算像素差异数量
        diff_count = np.sum(np.any(original_image != image_array, axis=2))
        total_pixels = 200 * 200
        diff_percentage = (diff_count / total_pixels) * 100
        
        print(f"\n图像比较结果:")
        print(f"总像素数: {total_pixels}")
        print(f"不同像素数: {diff_count}")
        print(f"差异百分比: {diff_percentage:.2f}%")
        
        if diff_count == 0:
            print("验证结果: 完全匹配！二进制数据正确地表示了原始图像。")
        else:
            print("验证结果: 存在差异。可能是由于图像格式转换或二进制数据处理过程中的问题。")
            
            # 创建差异图像
            diff_image = np.zeros((200, 200, 3), dtype=np.uint8)
            diff_mask = np.any(original_image != image_array, axis=2)
            diff_image[diff_mask] = [255, 0, 0]  # 标记差异像素为红色
            
            plt.figure(figsize=(8, 8))
            plt.imshow(diff_image)
            plt.title('差异图像 (红色表示不同的像素)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'difference.png'), dpi=300, bbox_inches='tight')
            # 不使用plt.show()，因为使用了Agg后端
    else:
        print(f"\n无法直接比较图像: 尺寸不匹配")
        print(f"原始图像尺寸: {original_image.shape}")
        print(f"重建图像尺寸: {image_array.shape}")
else:
    # 不使用plt.show()，因为使用了Agg后端
    print(f"\n原始处理后的图像文件不存在: {original_image_path}")
    print("无法进行图像比较")

print(f"\n验证结果已保存到: {output_dir}")

# 保存为BMP文件的详细信息
bmp_path = os.path.join(output_dir, 'reconstructed_image.bmp')
print(f"重建的BMP图像已保存为: {bmp_path}")

# 创建一个简单的BMP文件头信息函数
def get_bmp_info(bmp_path):
    with open(bmp_path, 'rb') as f:
        # 读取BMP文件头
        header = f.read(54)
        
        # 解析文件头信息
        file_size = struct.unpack('<I', header[2:6])[0]
        width = struct.unpack('<i', header[18:22])[0]
        height = struct.unpack('<i', header[22:26])[0]
        bit_depth = struct.unpack('<H', header[28:30])[0]
        compression = struct.unpack('<I', header[30:34])[0]
        image_size = struct.unpack('<I', header[34:38])[0]
        
        return {
            'file_size': file_size,
            'width': width,
            'height': height,
            'bit_depth': bit_depth,
            'compression': compression,
            'image_size': image_size
        }

# 显示BMP文件信息
try:
    bmp_info = get_bmp_info(bmp_path)
    print("\nBMP文件信息:")
    print(f"文件大小: {bmp_info['file_size']} 字节")
    print(f"图像宽度: {bmp_info['width']} 像素")
    print(f"图像高度: {bmp_info['height']} 像素")
    print(f"位深度: {bmp_info['bit_depth']} 位/像素")
    print(f"压缩方式: {bmp_info['compression']}")
    print(f"图像数据大小: {bmp_info['image_size']} 字节")
except Exception as e:
    print(f"读取BMP文件信息时出错: {e}")