import pandas as pd
import numpy as np

# 加载匹配结果
result_df = pd.read_excel("Result_Combined_Improved.xlsx")

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

# 分析两种方法的贡献
title_method = result_df[result_df['ID'] != "未找到"]['Method_Used'].value_counts().get('Title', 0)
device_method = result_df[result_df['ID'] != "未找到"]['Method_Used'].value_counts().get('Device', 0)

print(f"\n匹配方法贡献分析:")
print(f"标题方法贡献：{title_method}条 ({(title_method/matched)*100:.2f}%)")
print(f"机型数据方法贡献：{device_method}条 ({(device_method/matched)*100:.2f}%)")

# 分析两种方法的分数对比
print("\n两种方法的平均分数:")
mean_title = result_df['Title_Score'].mean()
mean_device = result_df['Device_Score'].mean()
print(f"标题方法平均分数: {mean_title:.2f}")
print(f"机型数据方法平均分数: {mean_device:.2f}")

# 显示几个匹配样例
print("\n匹配示例:")
matched_samples = result_df[result_df['ID'] != "未找到"].sample(min(5, matched))
for i, row in matched_samples.iterrows():
    print(f"原始标题: {row['pro_title']}")
    print(f"存储信息: {row['pro_model']}")
    print(f"机型数据: {row['机型数据']}")
    print(f"清理后(标题): {row['Cleaned_Title']} (分数: {row['Title_Score']})")
    print(f"清理后(机型): {row['Cleaned_Device']} (分数: {row['Device_Score']})")
    print(f"最终匹配: {row['Final_Match']} (使用方法: {row['Method_Used']})")
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
    print(f"清理后(标题): {row['Cleaned_Title']} (分数: {row['Title_Score']})")
    print(f"清理后(机型): {row['Cleaned_Device']} (分数: {row['Device_Score']})")
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
        elif 'xiaomi' in model_name or '小米' in model_name or 'redmi' in model_name or 'k' in model_name:
            return '小米/红米'
        elif 'honor' in model_name or '荣耀' in model_name:
            return '荣耀'
        elif 'oppo' in model_name or 'reno' in model_name:
            return 'OPPO'
        elif 'vivo' in model_name:
            return 'vivo'
        else:
            return '其他'
    
    matched_df = result_df[result_df['ID'] != "未找到"].copy()
    matched_df['Brand'] = matched_df['Final_Match'].apply(extract_brand)
    brand_counts = matched_df['Brand'].value_counts()
    print(brand_counts)
    
    # 统计每个品牌的总数量
    def count_brand(title, brand_name):
        title = str(title).lower()
        if brand_name == 'Apple' and ('iphone' in title or 'ipad' in title or 'apple' in title):
            return True
        elif brand_name == '华为' and ('huawei' in title or '华为' in title):
            return True
        elif brand_name == '小米/红米' and ('xiaomi' in title or '小米' in title or 'redmi' in title or '红米' in title):
            return True
        elif brand_name == '荣耀' and ('honor' in title or '荣耀' in title):
            return True
        elif brand_name == 'OPPO' and 'oppo' in title:
            return True
        elif brand_name == 'vivo' and 'vivo' in title:
            return True
        return False
    
    # 计算每个品牌的匹配率
    print("\n各品牌匹配率:")
    for brand in brand_counts.index:
        if brand == '其他':
            continue
        brand_total = sum(result_df['pro_title'].apply(lambda x: count_brand(x, brand)))
        if brand_total > 0:
            print(f"{brand}: {brand_counts[brand]}/{brand_total} ({brand_counts[brand]/brand_total*100:.2f}%)") 