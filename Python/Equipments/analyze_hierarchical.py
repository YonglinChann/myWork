import pandas as pd
import numpy as np

# 加载匹配结果
result_df = pd.read_excel("Result_Hierarchical.xlsx")

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

# 分析品牌检测结果
print("\n品牌检测统计:")
brand_counts = result_df['Detected_Brand'].value_counts()
print(brand_counts)

# 计算各品牌的匹配成功率
print("\n各品牌匹配成功率:")
for brand in brand_counts.index:
    if brand in ["未知品牌", "找不到品牌对应的机型", "无法确定机型"]:
        continue
    
    brand_total = len(result_df[result_df['Detected_Brand'] == brand])
    brand_matched = len(result_df[(result_df['Detected_Brand'] == brand) & (result_df['ID'] != "未找到")])
    
    if brand_total > 0:
        print(f"{brand}: {brand_matched}/{brand_total} ({brand_matched/brand_total*100:.2f}%)")

# 分析未匹配的原因
print("\n未匹配原因分析:")
unmatched_reasons = result_df[result_df['ID'] == "未找到"]['Detected_Brand'].value_counts()
print(unmatched_reasons)

# 显示几个匹配样例
print("\n匹配示例:")
matched_samples = result_df[result_df['ID'] != "未找到"].sample(min(5, matched))
for i, row in matched_samples.iterrows():
    print(f"原始标题: {row['pro_title']}")
    print(f"原始品牌: {row['pro_brand']}")
    print(f"原始型号: {row['pro_model']}")
    print(f"清理后品牌: {row['Cleaned_Brand']}")
    print(f"检测到品牌: {row['Detected_Brand']}")
    print(f"清理后标题: {row['Cleaned_Title']}")
    print(f"清理后型号: {row['Cleaned_Model']}")
    print(f"匹配到: {row['Matched_Key']}")
    print(f"ID: {row['ID']}")
    print(f"匹配分数: {row['Match_Score']}")
    print("-" * 50)

# 显示几个未匹配样例
print("\n未匹配示例:")
unmatched_samples = result_df[result_df['ID'] == "未找到"].sample(min(5, total - matched))
for i, row in unmatched_samples.iterrows():
    print(f"原始标题: {row['pro_title']}")
    print(f"原始品牌: {row['pro_brand']}")
    print(f"原始型号: {row['pro_model']}")
    print(f"清理后品牌: {row['Cleaned_Brand']}")
    print(f"检测到品牌: {row['Detected_Brand']}")
    print(f"清理后标题: {row['Cleaned_Title']}")
    print(f"清理后型号: {row['Cleaned_Model']}")
    print(f"未匹配原因: {row['Detected_Brand'] if row['Detected_Brand'] in ['未知品牌', '找不到品牌对应的机型', '无法确定机型'] else '其他原因'}")
    print("-" * 50) 