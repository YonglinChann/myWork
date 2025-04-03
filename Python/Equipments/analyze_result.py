import pandas as pd

# 加载匹配结果
result_df = pd.read_excel("Result.xlsx")

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

# 显示几个匹配和未匹配的例子
print("\n匹配示例:")
matched_samples = result_df[result_df['ID'] != "未找到"].sample(min(5, matched))
for i, row in matched_samples.iterrows():
    print(f"原始标题: {row['pro_title']}")
    print(f"清理后: {row['Cleaned_Model']}")
    print(f"匹配到: {row['Matched_Model']}")
    print(f"ID: {row['ID']}")
    print(f"匹配分数: {row['Match_Score']}")
    print("-" * 50)

print("\n未匹配示例:")
unmatched_samples = result_df[result_df['ID'] == "未找到"].sample(min(5, total - matched))
for i, row in unmatched_samples.iterrows():
    print(f"原始标题: {row['pro_title']}")
    print(f"清理后: {row['Cleaned_Model']}")
    print("-" * 50) 