import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re

print("正在加载和预处理数据...")

# 加载Excel文件
model_df = pd.read_excel("机型表.xlsx")  # 机型表
data_df = pd.read_excel("新合并.xlsx")   # 源数据

# 确保所有值都是字符串类型
model_df['Models'] = model_df['Models'].astype(str)
model_df['Deploy'] = model_df['Deploy'].astype(str)
if 'Brand' in model_df.columns:
    model_df['Brand'] = model_df['Brand'].astype(str)
else:
    # 如果机型表没有Brand列，尝试从Models提取品牌
    print("机型表中没有Brand列，尝试从Models提取品牌...")
    def extract_brand_from_model(model_name):
        model_name = model_name.lower()
        if 'iphone' in model_name or 'ipad' in model_name:
            return 'Apple'
        elif 'huawei' in model_name or '华为' in model_name:
            return '华为'
        elif 'xiaomi' in model_name or '小米' in model_name:
            return '小米'
        elif 'redmi' in model_name or '红米' in model_name:
            return '红米'
        elif 'honor' in model_name or '荣耀' in model_name:
            return '荣耀'
        elif 'oppo' in model_name or 'reno' in model_name:
            return 'OPPO'
        elif 'vivo' in model_name or 'iqoo' in model_name:
            return 'vivo'
        else:
            return '其他'
    
    model_df['Brand'] = model_df['Models'].apply(extract_brand_from_model)

# 预处理源数据
for col in ['pro_title', 'pro_model', 'pro_brand', '机型数据']:
    if col in data_df.columns:
        data_df[col] = data_df[col].fillna('')
        data_df[col] = data_df[col].astype(str)

# 在机型表中创建一个查找键
model_df['Lookup_Key'] = model_df['Models'] + " " + model_df['Deploy']

# 标准化品牌名称
def normalize_brand(brand):
    brand = str(brand).lower().strip()
    brand_mapping = {
        'apple': 'Apple',
        'iphone': 'Apple',
        'ipad': 'Apple',
        'huawei': '华为',
        '华为': '华为',
        'xiaomi': '小米',
        '小米': '小米',
        'redmi': '红米',
        '红米': '红米',
        'honor': '荣耀',
        '荣耀': '荣耀',
        'oppo': 'OPPO',
        'reno': 'OPPO',
        'vivo': 'vivo',
        'iqoo': 'vivo'
    }
    
    for key, value in brand_mapping.items():
        if key in brand:
            return value
    
    return brand

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

# 清理文本的通用函数
def clean_text(text):
    if not text or text == 'nan':
        return ""
    
    text = str(text).lower()
    
    # 移除常见的广告词和无关文本
    phrases_to_remove = [
        "全新国行", "租物高分专享", "[非监管机]", "[不上征信]", 
        "租满6期支持归还", "正品", "现货", "全新", "国行",
        "官方", "授权", "专卖", "直营", "旗舰店", "国内", "原封", 
        "未激活", "已验机", "全新未拆封", "【全新】", "（全新）", "(全新)",
        "融租-", "租满", "租期", "天起租", "支持归还", "无需归还"
    ]
    
    # 清理标题
    cleaned = text
    for phrase in phrases_to_remove:
        cleaned = cleaned.replace(phrase.lower(), "")
    
    # 移除括号内的内容
    cleaned = re.sub(r'\([^)]*\)', '', cleaned)
    cleaned = re.sub(r'\（[^）]*\）', '', cleaned)
    cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
    
    # 移除多余的空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

# 清理品牌信息
data_df['Cleaned_Brand'] = data_df['pro_brand'].apply(lambda x: normalize_brand(clean_text(x)))

# 清理标题信息
data_df['Cleaned_Title'] = data_df['pro_title'].apply(clean_text)

# 清理型号信息
data_df['Cleaned_Model'] = data_df['pro_model'].apply(clean_text)

# 分层匹配函数
def hierarchical_match(row, model_df):
    # 第一步：匹配品牌
    brand = row['Cleaned_Brand']
    
    # 如果品牌为空，尝试从标题提取
    if not brand:
        title = row['Cleaned_Title']
        if 'iphone' in title or 'ipad' in title or 'apple' in title:
            brand = 'Apple'
        elif 'huawei' in title or '华为' in title:
            brand = '华为'
        elif 'xiaomi' in title or '小米' in title:
            brand = '小米'
        elif 'redmi' in title or '红米' in title:
            brand = '红米'
        elif 'honor' in title or '荣耀' in title:
            brand = '荣耀'
        elif 'oppo' in title or 'reno' in title:
            brand = 'OPPO'
        elif 'vivo' in title or 'iqoo' in title:
            brand = 'vivo'
    
    # 如果仍然无法确定品牌，则无法继续匹配
    if not brand or brand == '其他':
        return None, 0, "未知品牌"
    
    # 筛选特定品牌的所有型号
    brand_models = model_df[model_df['Brand'] == brand]
    
    if len(brand_models) == 0:
        return None, 0, "找不到品牌对应的机型"
    
    # 第二步：匹配机型（Models）
    title = row['Cleaned_Title']
    matched_model = None
    best_score = 0
    
    for idx, model_row in brand_models.iterrows():
        model_name = model_row['Models'].lower()
        
        # 使用不同匹配算法尝试匹配
        score1 = fuzz.token_sort_ratio(title, model_name) * 1.0
        score2 = fuzz.partial_ratio(title, model_name) * 0.9
        score3 = fuzz.token_set_ratio(title, model_name) * 0.85
        
        score = max(score1, score2, score3)
        
        if score > best_score:
            best_score = score
            matched_model = model_row
    
    # 如果机型匹配分数太低，则无法确定具体机型
    if best_score < 60:
        return None, 0, "无法确定机型"
    
    # 第三步：匹配具体型号（Deploy）
    deploy = row['Cleaned_Model']
    matched_deploy = matched_model['Deploy'].lower()
    
    deploy_score = fuzz.token_sort_ratio(deploy, matched_deploy)
    
    # 最终得分是机型匹配和型号匹配的加权平均
    final_score = (best_score * 0.7) + (deploy_score * 0.3)
    
    return matched_model['Lookup_Key'], final_score, brand

print("正在进行分层匹配...")
# 应用分层匹配
results = []
scores = []
brands = []

total = len(data_df)
for i, row in data_df.iterrows():
    if i % 500 == 0:
        print(f"处理进度: {i}/{total}")
    
    match, score, brand = hierarchical_match(row, model_df)
    results.append(match)
    scores.append(score)
    brands.append(brand)

data_df['Matched_Key'] = results
data_df['Match_Score'] = scores
data_df['Detected_Brand'] = brands

# 合并以获取ID
print("正在合并结果...")
result_df = data_df.merge(
    model_df[['Lookup_Key', 'ID']],
    left_on='Matched_Key',
    right_on='Lookup_Key',
    how='left'
)

# 选择所需列
result_columns = ['pro_title', 'pro_brand', 'pro_model', '机型数据', 
                 'Cleaned_Brand', 'Cleaned_Title', 'Cleaned_Model', 
                 'Detected_Brand', 'Matched_Key', 'Match_Score', 'ID']
                 
result_df = result_df[result_columns]
result_df['ID'] = result_df['ID'].fillna("未找到")

# 保存到新的Excel文件
result_df.to_excel("Result_Hierarchical.xlsx", index=False)
print("匹配完成。输出已保存到 'Result_Hierarchical.xlsx'。")

# 输出匹配统计信息
total = len(result_df)
matched = result_df['ID'].ne("未找到").sum()
print(f"\n匹配统计：")
print(f"总记录数：{total}")
print(f"成功匹配数：{matched}")
print(f"匹配率：{(matched/total)*100:.2f}%")

# 品牌分布统计
if matched > 0:
    print("\n匹配成功的设备品牌分布:")
    brand_counts = result_df[result_df['ID'] != "未找到"]['Detected_Brand'].value_counts()
    print(brand_counts)
    
    print("\n各品牌匹配率:")
    for brand, count in brand_counts.items():
        brand_total = len(result_df[result_df['Detected_Brand'] == brand])
        print(f"{brand}: {count}/{brand_total} ({count/brand_total*100:.2f}%)") 