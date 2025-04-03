import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

# 加载Excel文件
model_df = pd.read_excel("机型表.xlsx")  # 机型表
data_df = pd.read_excel("新合并.xlsx")   # 源数据

# 确保所有值都是字符串类型
model_df['Models'] = model_df['Models'].astype(str)
model_df['Deploy'] = model_df['Deploy'].astype(str)

# 在机型表中创建一个查找键
model_df['Lookup_Key'] = model_df['Models'] + " " + model_df['Deploy']

# 处理存储容量的函数
def normalize_storage(storage_str):
    storage_str = str(storage_str).lower()
    
    # 标准化存储单位
    storage_mapping = {
        'g': 'gb',
        't': 'tb'
    }
    
    for old, new in storage_mapping.items():
        if storage_str.endswith(old) and not storage_str.endswith(new):
            storage_str = storage_str[:-1] + new
    
    # 处理常见的存储格式 如 "16+512gb" 变为 "16gb+512gb"
    storage_pattern = r'(\d+)\+(\d+)(gb|tb)'
    match = re.search(storage_pattern, storage_str)
    if match:
        ram, rom, unit = match.groups()
        storage_str = f"{ram}{unit}+{rom}{unit}"
    
    return storage_str

# 清理产品标题的函数
def clean_title(title, storage):
    # 确保输入是字符串类型
    title = str(title)
    storage = str(storage)
    
    # 移除常见的广告词和无关文本
    phrases_to_remove = [
        "全新国行", "租物高分专享", "[非监管机]", "[不上征信]", 
        "租满6期支持归还", "正品", "现货", "全新", "国行",
        "官方", "授权", "专卖", "直营", "旗舰店", "国内", "原封", 
        "未激活", "已验机", "全新未拆封", "【全新】", "（全新）", "(全新)",
        "融租-", "租满", "租期", "天起租", "支持归还", "无需归还"
    ]
    
    # 清理标题
    cleaned = title
    for phrase in phrases_to_remove:
        cleaned = cleaned.replace(phrase, "")
    
    # 移除括号内的内容，如 "iPhone 14(不要动数据！！！)"
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\（[^）]*\）', '', cleaned)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    
    # 处理特殊格式的机型名称
    if "iphone" in cleaned.lower() or "华为" in cleaned or "荣耀" in cleaned or "小米" in cleaned:
        # 尝试提取更准确的型号
        model_patterns = [
            r'(iphone\s*\d+\s*(pro|plus|max|mini)?)', # iPhone 型号
            r'(华为\s*[a-zA-Z0-9]+\s*\d+(\s*pro)?)', # 华为型号
            r'(荣耀\s*[a-zA-Z0-9]+\s*\d+(\s*pro)?)', # 荣耀型号
            r'(小米\s*\d+(\s*[a-zA-Z]+)?)', # 小米型号
        ]
        
        for pattern in model_patterns:
            match = re.search(pattern, cleaned.lower())
            if match and match.group(1):
                cleaned = match.group(1)
                break
    
    # 移除多余的空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 标准化大小写
    cleaned = cleaned.lower()
    
    # 添加存储信息
    storage = normalize_storage(storage)
    return (cleaned.strip() + " " + storage).strip()

# 对源数据应用清理
data_df['Cleaned_Model'] = data_df.apply(
    lambda row: clean_title(row['pro_title'], row['pro_model']), axis=1
)

# 标准化机型表的关键字
model_df['Normalized_Key'] = model_df['Lookup_Key'].str.lower()
model_df['Normalized_Key'] = model_df['Normalized_Key'].apply(normalize_storage)

# 查找最佳匹配的函数
def get_best_match(cleaned_model, choices, lookup_dict, threshold=70):
    # 确保输入是字符串类型
    cleaned_model = str(cleaned_model).lower()
    
    # 尝试多种匹配算法
    matchers = [
        (fuzz.token_sort_ratio, 1.0),
        (fuzz.partial_ratio, 0.9),
        (fuzz.token_set_ratio, 0.85),
        (fuzz.ratio, 0.8)
    ]
    
    best_match = None
    best_score = 0
    
    # 首先尝试精确匹配
    if cleaned_model in lookup_dict:
        return lookup_dict[cleaned_model]
    
    # 针对特定类型的设备使用不同的匹配阈值
    device_type_thresholds = {
        'iphone': 65,
        'ipad': 65,
        'huawei': 65,
        'xiaomi': 65,
        'honor': 65,
        'oppo': 65,
        'vivo': 65
    }
    
    # 检查是否为特定设备类型并调整阈值
    for device, device_threshold in device_type_thresholds.items():
        if device in cleaned_model:
            threshold = device_threshold
            break
    
    # 尝试不同的匹配算法
    for matcher, weight in matchers:
        match, score = process.extractOne(cleaned_model, choices, scorer=matcher)
        weighted_score = score * weight
        
        if weighted_score > best_score:
            best_match = match
            best_score = weighted_score
    
    if best_score >= threshold:
        return best_match
    return None

# 创建查找字典以加快精确匹配
lookup_dict = {key.lower(): key for key in model_df['Lookup_Key'].tolist()}

# 进行模糊匹配
choices = model_df['Lookup_Key'].tolist()
data_df['Matched_Model'] = data_df['Cleaned_Model'].apply(
    lambda x: get_best_match(x, choices, lookup_dict)
)

# 合并以获取ID
result_df = data_df.merge(
    model_df[['Lookup_Key', 'ID']],
    left_on='Matched_Model',
    right_on='Lookup_Key',
    how='left'
)

# 选择相关列并重命名
result_df = result_df[data_df.columns.tolist() + ['ID']]
result_df['ID'] = result_df['ID'].fillna("未找到")

# 添加匹配度信息
result_df['Cleaned_Model'] = result_df['Cleaned_Model'].str.lower()
result_df['Matched_Model'] = result_df['Matched_Model'].str.lower()
result_df['Match_Score'] = result_df.apply(
    lambda row: fuzz.token_sort_ratio(row['Cleaned_Model'], row['Matched_Model']) 
    if pd.notna(row['Matched_Model']) else 0, 
    axis=1
)

# 保存到新的Excel文件
result_df.to_excel("Result.xlsx", index=False)
print("匹配完成。输出已保存到 'Result.xlsx'。")

# 输出匹配统计信息
total = len(result_df)
matched = result_df['ID'].ne("未找到").sum()
print(f"\n匹配统计：")
print(f"总记录数：{total}")
print(f"成功匹配数：{matched}")
print(f"匹配率：{(matched/total)*100:.2f}%")