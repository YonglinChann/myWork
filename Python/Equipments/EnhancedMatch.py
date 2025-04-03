import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

# 加载Excel文件
model_df = pd.read_excel("机型表.xlsx")  # 机型表
data_df = pd.read_excel("新合并.xlsx")   # 源数据

# 确保"Combined"列为字符串类型
model_df['Combined'] = model_df['Combined'].astype(str)

# 品牌同义词映射表
brand_synonyms = {
    "apple": ["apple", "苹果", "iphone", "ipad", "macbook", "airpods", "watch"],
    "huawei": ["huawei", "华为", "荣耀", "honor", "mate", "nova", "p系列"], 
    "xiaomi": ["xiaomi", "小米", "红米", "redmi", "poco"],
    "oppo": ["oppo", "一加", "oneplus", "realme"],
    "vivo": ["vivo", "iqoo"],
    "samsung": ["samsung", "三星", "galaxy"],
    "microsoft": ["microsoft", "微软", "surface"],
    "sony": ["sony", "索尼", "xperia"],
    "nokia": ["nokia", "诺基亚"],
    "motorola": ["motorola", "摩托罗拉", "moto"],
    "zte": ["zte", "中兴"],
    "lenovo": ["lenovo", "联想"],
    "meizu": ["meizu", "魅族"]
}

# 品牌名称标准化
def normalize_brand(text):
    text_lower = text.lower()
    for standard_brand, synonyms in brand_synonyms.items():
        for synonym in synonyms:
            if synonym in text_lower:
                # 特殊处理：对于Apple产品，保留原有产品线前缀(iPhone, iPad等)
                if standard_brand == "apple" and any(prefix in text_lower for prefix in ["iphone", "ipad", "macbook", "airpods", "watch"]):
                    continue
                # 其他情况，替换为标准品牌名
                return text.replace(synonym, standard_brand)
    return text

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
    
    # 精确提取设备型号的正则表达式
    precise_patterns = {
        # iPhone型号 - 更精确地匹配完整型号系列
        'iphone': [
            r'(iphone\s*\d+\s*pro\s*max\s*(\d+\s*g[bt])?)',  # iPhone Pro Max系列
            r'(iphone\s*\d+\s*pro\s*(\d+\s*g[bt])?)',        # iPhone Pro系列
            r'(iphone\s*\d+\s*plus\s*(\d+\s*g[bt])?)',       # iPhone Plus系列
            r'(iphone\s*\d+\s*mini\s*(\d+\s*g[bt])?)',       # iPhone Mini系列
            r'(iphone\s*\d+\s*(\d+\s*g[bt])?)'               # 标准iPhone系列
        ],
        # 华为型号 - 处理各子系列
        'huawei': [
            r'(华为|huawei)\s*(mate|p|nova)\s*\d+\s*(pro|plus|max)?\s*(折叠|4g|5g)?',
        ],
        # 小米型号
        'xiaomi': [
            r'(小米|xiaomi)\s*\d+\s*(pro|plus|ultra)?\s*(折叠|4g|5g)?',
            r'(红米|redmi)\s*([a-z]+)?\s*\d+\s*(pro|plus)?'
        ],
        # 其他型号可以继续添加...
    }
    
    found_match = False
    # 检查是否是特定品牌设备
    for brand, patterns in precise_patterns.items():
        if brand in cleaned.lower():
            # 尝试所有该品牌的精确匹配模式
            for pattern in patterns:
                match = re.search(pattern, cleaned.lower())
                if match and match.group(0):
                    cleaned = match.group(0)
                    found_match = True
                    break
            if found_match:
                break
    
    # 如果没有找到精确匹配，使用旧的提取方法
    if not found_match:
        # 处理特殊格式的机型名称
        generic_patterns = [
            r'(iphone\s*\d+\s*(pro|plus|max|mini)?)', # iPhone 型号
            r'(华为\s*[a-zA-Z0-9]+\s*\d+(\s*pro)?)', # 华为型号
            r'(荣耀\s*[a-zA-Z0-9]+\s*\d+(\s*pro)?)', # 荣耀型号
            r'(小米\s*\d+(\s*[a-zA-Z]+)?)', # 小米型号
        ]
        
        for pattern in generic_patterns:
            match = re.search(pattern, cleaned.lower())
            if match and match.group(1):
                cleaned = match.group(1)
                break
    
    # 移除多余的空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 标准化大小写
    cleaned = cleaned.lower()
    
    # 标准化品牌名称
    cleaned = normalize_brand(cleaned)
    
    # 添加存储信息
    storage = normalize_storage(storage)
    return (cleaned.strip() + " " + storage).strip()

# 对源数据应用清理
data_df['Cleaned_Model'] = data_df.apply(
    lambda row: clean_title(row['pro_title'], row['pro_model']), axis=1
)

# 标准化机型表的关键字
model_df['Normalized_Key'] = model_df['Combined'].str.lower()
model_df['Normalized_Key'] = model_df['Normalized_Key'].apply(normalize_storage)
model_df['Normalized_Key'] = model_df['Normalized_Key'].apply(normalize_brand)

# 更智能的最佳匹配函数，避免部分匹配问题
def get_best_match(cleaned_model, choices, lookup_dict, threshold=70):
    # 确保输入是字符串类型
    cleaned_model = str(cleaned_model).lower()
    
    # 尝试精确匹配
    if cleaned_model in lookup_dict:
        return lookup_dict[cleaned_model]
    
    # 获取潜在匹配列表
    potential_matches = []
    
    # 尝试多种匹配算法，为可能的候选项评分
    matchers = [
        (fuzz.token_sort_ratio, 1.0),
        (fuzz.token_set_ratio, 0.95),
        (fuzz.partial_ratio, 0.9),
        (fuzz.ratio, 0.85)
    ]
    
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
    
    # 获取多个候选匹配项
    for matcher, weight in matchers:
        # 获取前5个最佳匹配
        matches = process.extract(cleaned_model, choices, 
                                  scorer=matcher, limit=5)
        
        for match, score in matches:
            weighted_score = score * weight
            if weighted_score >= threshold:
                potential_matches.append((match, weighted_score))
    
    # 如果没有匹配项达到阈值
    if not potential_matches:
        return None
    
    # 按分数排序
    potential_matches.sort(key=lambda x: x[1], reverse=True)
    
    # 解决部分匹配问题：如果最高分的几个结果差异不大，优先选择更完整/更长的匹配
    top_matches = [m for m in potential_matches if m[1] >= potential_matches[0][1] * 0.95]
    
    if len(top_matches) > 1:
        # 倾向于匹配更完整的型号 (更长的字符串，特别是包含"pro max"等完整系列名称)
        # 为匹配项评分：字符串长度 + 关键词权重
        keyword_weights = {"pro max": 5, "ultra": 4, "pro": 3, "plus": 2, "max": 2, "mini": 1}
        
        match_scores = []
        for match, score in top_matches:
            match_lower = match.lower()
            length_score = len(match)  # 字符串长度作为基础分
            keyword_score = 0
            
            # 添加关键词权重
            for kw, weight in keyword_weights.items():
                if kw in match_lower:
                    keyword_score += weight
            
            # 如果匹配的字符串完全包含查询字符串，给予额外加分
            if cleaned_model in match_lower:
                keyword_score += 3
                
            match_scores.append((match, score, length_score + keyword_score))
        
        # 按综合评分排序
        match_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        return match_scores[0][0]
    
    # 如果只有一个最佳匹配，直接返回
    return potential_matches[0][0]

# 创建查找字典以加快精确匹配
lookup_dict = {key.lower(): key for key in model_df['Combined'].tolist()}

# 进行模糊匹配
choices = model_df['Combined'].tolist()
data_df['Matched_Model'] = data_df['Cleaned_Model'].apply(
    lambda x: get_best_match(x, choices, lookup_dict)
)

# 合并以获取ID
result_df = data_df.merge(
    model_df[['Combined', 'ID']],
    left_on='Matched_Model',
    right_on='Combined',
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
result_df.to_excel("EnhancedResult_Combined.xlsx", index=False)
print("增强匹配完成。输出已保存到 'EnhancedResult_Combined.xlsx'。")

# 输出匹配统计信息
total = len(result_df)
matched = result_df['ID'].ne("未找到").sum()
print(f"\n匹配统计：")
print(f"总记录数：{total}")
print(f"成功匹配数：{matched}")
print(f"匹配率：{(matched/total)*100:.2f}%") 