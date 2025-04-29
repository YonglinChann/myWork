# 处理墨水屏HEX数据文件，提取数据包内容并生成列表格式

import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 输入和输出文件路径
input_file = os.path.join(current_dir, 'output', 'eink_display_data_hex.txt')
output_file = os.path.join(current_dir, 'output', 'eink_display_data_list.py')

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

# 主函数
def main():
    print(f"正在处理HEX数据文件: {input_file}")
    hex_packets = process_hex_data(input_file)
    print(f"提取了 {len(hex_packets)} 个数据包")
    
    save_as_python_list(hex_packets, output_file)
    print(f"已将处理结果保存到: {output_file}")

if __name__ == "__main__":
    main()