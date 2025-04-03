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

# 处理NaN值
data_df['机型数据'] = data_df['机型数据'].fillna('')
data_df['机型数据'] = data_df['机型数据'].astype(str)

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

# 清理产品数据的函数
def clean_device_data(device_data):
    if not device_data or device_data == 'nan':
        return ""
        
    # 确保输入是字符串类型
    device_data = str(device_data).lower()
    
    # 移除常见的广告词和无关文本
    phrases_to_remove = [
        "全新国行", "租物高分专享", "[非监管机]", "[不上征信]", 
        "租满6期支持归还", "正品", "现货", "全新", "国行",
        "官方", "授权", "专卖", "直营", "旗舰店", "国内", "原封", 
        "未激活", "已验机", "全新未拆封", "【全新】", "（全新）", "(全新)",
        "融租-", "租满", "租期", "天起租", "支持归还", "无需归还", "融租"
    ]
    
    # 清理数据
    cleaned = device_data
    for phrase in phrases_to_remove:
        cleaned = cleaned.replace(phrase.lower(), "")
    
    # 移除括号内的内容
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\（[^）]*\）', '', cleaned)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    
    # 品牌名称规范化
    brand_mapping = {
        'apple': 'apple ',
        'huawei': '华为 ',
        'xiaomi': '小米 ',
        'honor': '荣耀 ',
        'oppo': 'oppo ',
        'vivo': 'vivo '
    }
    
    # 提取设备型号
    extracted_model = ""
    
    # 提取iPhone型号
    if "iphone" in cleaned:
        iphone_pattern = r'(iphone\s*\d+\s*(pro|plus|max|mini)?)'
        match = re.search(iphone_pattern, cleaned)
        if match:
            extracted_model = match.group(1)
            
            # 提取存储容量
            storage_pattern = r'(\d+)\s*(gb|g|tb|t)'
            storage_match = re.search(storage_pattern, cleaned)
            if storage_match:
                storage = storage_match.group(0)
                extracted_model += " " + normalize_storage(storage)
    
    # 提取iPad型号
    elif "ipad" in cleaned:
        ipad_pattern = r'(ipad\s*(pro|air|mini)?\s*\d*\.?\d*[\'"]?)'
        match = re.search(ipad_pattern, cleaned)
        if match:
            extracted_model = match.group(1)
            
            # 提取存储容量
            storage_pattern = r'(\d+)\s*(gb|g|tb|t)'
            storage_match = re.search(storage_pattern, cleaned)
            if storage_match:
                storage = storage_match.group(0)
                extracted_model += " " + normalize_storage(storage)
    
    # 对于其他品牌，尝试提取型号+存储信息
    else:
        # 通用模式：品牌名 + 型号数字 + 可能的后缀 + 存储
        model_pattern = r'(华为|荣耀|小米|oppo|vivo|一加)?\s*([a-zA-Z0-9]+\s*\d+\s*(pro|plus|max|ultra|青春版)?)'
        match = re.search(model_pattern, cleaned)
        if match:
            if match.group(1):  # 如果匹配到品牌
                brand = match.group(1).strip()
                model = match.group(2).strip()
                extracted_model = f"{brand} {model}"
            else:
                extracted_model = match.group(2).strip()
            
            # 提取存储容量
            storage_pattern = r'(\d+)\s*(gb|g|tb|t)'
            storage_match = re.search(storage_pattern, cleaned)
            if storage_match:
                storage = storage_match.group(0)
                extracted_model += " " + normalize_storage(storage)
    
    # 如果提取失败，使用清理后的原始文本
    if not extracted_model:
        # 移除多余的空格
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()
    
    # 移除多余的空格
    extracted_model = re.sub(r'\s+', ' ', extracted_model)
    return extracted_model.strip()

# 对源数据应用清理
data_df['Cleaned_Model'] = data_df['机型数据'].apply(clean_device_data)

# 标准化机型表的关键字
model_df['Normalized_Key'] = model_df['Lookup_Key'].str.lower()
model_df['Normalized_Key'] = model_df['Normalized_Key'].apply(normalize_storage)

# 查找最佳匹配的函数
def get_best_match(cleaned_model, choices, lookup_dict, threshold=70):
    if not cleaned_model:
        return None
        
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

# 选择所有原始列并添加匹配信息
result_columns = data_df.columns.tolist() + ['ID']
result_df = result_df[result_columns]
result_df['ID'] = result_df['ID'].fillna("未找到")

# 添加匹配度信息
result_df['Match_Score'] = result_df.apply(
    lambda row: fuzz.token_sort_ratio(str(row['Cleaned_Model']).lower(), str(row['Matched_Model']).lower()) 
    if pd.notna(row['Matched_Model']) else 0, 
    axis=1
)

# 保存到新的Excel文件
result_df.to_excel("Result_SingleColumn.xlsx", index=False)
print("匹配完成。输出已保存到 'Result_SingleColumn.xlsx'。")

# 输出匹配统计信息
total = len(result_df)
matched = result_df['ID'].ne("未找到").sum()
print(f"\n匹配统计：")
print(f"总记录数：{total}")
print(f"成功匹配数：{matched}")
print(f"匹配率：{(matched/total)*100:.2f}%") 