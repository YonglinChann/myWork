import pandas as pd
import numpy as np

# 加载匹配结果
result_df = pd.read_excel("Result_Combined.xlsx")

# 基本统计
total = len(result_df)
matched = result_df['ID'].ne("未找到").sum()
match_rate = (matched/total)*100

print(f"总记录数: {total}")
print(f"成功匹配数: {matched}")
print(f"匹配率: {match_rate:.2f}%")

# 分析匹配分数分布
if 'Match_Score' in result_df.columns:
    print("\n匹配分数分布:")
    score_bins = [0, 60, 70, 80, 90, 100]
    score_labels = ['0-60', '60-70', '70-80', '80-90', '90-100']
    score_dist = pd.cut(result_df['Match_Score'], bins=score_bins, labels=score_labels)
    print(score_dist.value_counts().sort_index())

# 加载中间结果以分析两种方法的贡献
try:
    # 尝试直接读取中间结果列
    title_better = (result_df['Title_Score'] >= result_df['Device_Score']).sum()
    device_better = (result_df['Title_Score'] < result_df['Device_Score']).sum()
    print(f"\n标题方法提供更好匹配的记录数: {title_better}")
    print(f"机型数据方法提供更好匹配的记录数: {device_better}")
except KeyError:
    # 如果中间结果列不存在，则加载原始数据重新计算
    print("\n注意：无法直接获取中间结果，无法计算每种方法的贡献")

# 显示几个匹配样例
print("\n匹配示例:")
matched_samples = result_df[result_df['ID'] != "未找到"].sample(min(5, matched))
for i, row in matched_samples.iterrows():
    print(f"原始标题: {row['pro_title']}")
    print(f"存储信息: {row['pro_model']}")
    print(f"机型数据: {row['机型数据']}")
    print(f"清理后(标题): {row['Cleaned_Title']}")
    print(f"清理后(机型): {row['Cleaned_Device']}")
    print(f"最终匹配: {row['Final_Match']}")
    print(f"ID: {row['ID']}")
    print(f"匹配分数: {row['Match_Score']}")
    print("-" * 50)

# 显示几个未匹配样例
print("\n未匹配示例:")
unmatched_samples = result_df[result_df['ID'] == "未找到"].sample(min(5, total - matched))
for i, row in unmatched_samples.iterrows():
    print(f"原始标题: {row['pro_title']}")
    print(f"存储信息: {row['pro_model']}")
    print(f"机型数据: {row['机型数据']}")
    print(f"清理后(标题): {row['Cleaned_Title']}")
    print(f"清理后(机型): {row['Cleaned_Device']}")
    print("-" * 50)

# 分析匹配成功的设备类型
if matched > 0:
    print("\n匹配成功的设备类型分布:")
    # 提取品牌信息（简单方法，可能不完全准确）
    def extract_brand(model_name):
        model_name = str(model_name).lower()
        if 'iphone' in model_name or 'ipad' in model_name:
            return 'Apple'
        elif 'huawei' in model_name or '华为' in model_name:
            return '华为'
        elif 'xiaomi' in model_name or '小米' in model_name:
            return '小米'
        elif 'honor' in model_name or '荣耀' in model_name:
            return '荣耀'
        elif 'oppo' in model_name:
            return 'OPPO'
        elif 'vivo' in model_name:
            return 'vivo'
        else:
            return '其他'
    
    matched_df = result_df[result_df['ID'] != "未找到"]
    matched_df['Brand'] = matched_df['Final_Match'].apply(extract_brand)
    brand_counts = matched_df['Brand'].value_counts()
    print(brand_counts) 