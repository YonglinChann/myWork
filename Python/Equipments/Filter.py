import pandas as pd
import os
import re

# 读取CSV文件
csv_path = '/Users/chenyonglin/myCode/gitee/myWork/Python/Equipments/机型参数数据20250324.csv'
df = pd.read_csv(csv_path)

# 定义需要移除的营销文本
marketing_terms = [
    '全新国行', '租物高分专享', '非监管机', '不上征信', '租满6期支持归还',
    '99新', '95新', '新品上新', '全新', '正品', '顺丰', '发货', '年度',
    '首月1元', '专区专享'
]

# 清理文本的函数
def clean_title(title):
    # 转换为字符串
    title = str(title)
    
    # 移除营销文本
    for term in marketing_terms:
        title = title.replace(term, '')
    
    # 移除方括号及其内容
    title = re.sub(r'\[.*?\]', '', title)
    
    # 移除破折号
    title = title.replace('-', '')
    
    # 清理多余的空格
    title = ' '.join(title.split())
    
    return title.strip()

# 应用清理函数到pro_title列
df['cleaned_title'] = df['pro_title'].apply(clean_title)

# 使用清理后的标题进行统计
title_counts = df.groupby('cleaned_title').agg({
    '数量': 'sum',  # 合并相同标题的数量
    'pro_title': lambda x: list(set(x))  # 保存原始标题列表
}).reset_index()

# 按数量降序排序
title_counts = title_counts.sort_values('数量', ascending=False)

# 打印结果
print("\n=== 设备型号统计（清理后）===")
print(f"共发现 {len(title_counts)} 个不同型号\n")

for index, row in title_counts.iterrows():
    print(f"清理后型号: {row['cleaned_title']}")
    print(f"原始型号: {row['pro_title']}")
    print(f"数量: {int(row['数量'])} 台")
    print("-" * 50)

# 将结果保存到Excel文件
output_path = '/Users/chenyonglin/myCode/gitee/myWork/Python/Equipments/device_titles.xlsx'

# 创建Excel写入器
with pd.ExcelWriter(output_path) as writer:
    # 写入清理后的统计数据
    title_counts.to_excel(writer, sheet_name='设备型号统计', index=False)
    
    # 添加带有清理后标题的原始数据sheet
    df_with_cleaned = df[['pro_title', 'cleaned_title', 'pro_brand', 'pro_model', 'pro_color', '数量']]
    df_with_cleaned.to_excel(writer, sheet_name='原始数据', index=False)

print(f"\n统计结果已保存到: {output_path}")