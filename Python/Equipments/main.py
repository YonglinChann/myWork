import pandas as pd
import re

# 读取CSV文件
df = pd.read_csv('/Users/chenyonglin/myCode/gitee/myWork/Python/Equipments/机型参数数据20250324.csv')

# 创建设备类型映射字典
device_type_mapping = {
    'Smartphone': '手机',
    'Tablet': '平板',
    'Computer': '电脑',
    'Smartwatch': '手表',
    'Earphone': '耳机',
    'Other': '其它'
}

# 创建品牌映射字典，将不同表示方式映射到统一名称
brand_mapping = {
    'Apple': 'Apple 苹果',
    '苹果': 'Apple 苹果',
    'apple': 'Apple 苹果',
    '华为': 'Huawei 华为',
    'HUAWEI': 'Huawei 华为',
    '华为（HUAWEI）': 'Huawei 华为',
    '小米': 'Xiaomi 小米',
    'Xiaomi': 'Xiaomi 小米',
    '小米（MI）': 'Xiaomi 小米',
    '红米': 'Xiaomi 小米',
    'Redmi': 'Xiaomi 小米',
    'vivo': 'VIVO',
    'OPPO': 'OPPO',
    'oppo': 'OPPO',
    '三星': 'Samsung 三星',
    '三星（SAMSUNG）': 'Samsung 三星',
    '荣耀': 'Honor 荣耀',
    '荣耀（HONOR）': 'Honor 荣耀',
    '魅族': 'Meizu 魅族',
    '努比亚': 'Nubia 努比亚',
    '努比亚（nubia）': 'Nubia 努比亚',
    '联想': 'Lenovo 联想',
    '联想（Lenovo）': 'Lenovo 联想',
    '华硕': 'Asus 华硕',
    '华硕（Asus）': 'Asus 华硕',
    '华硕（ASUS）': 'Asus 华硕',
    '华硕笔记本': 'Asus 华硕',
    '惠普': 'HP 惠普',
    '惠普（HP）': 'HP 惠普',
    '戴尔': 'Dell 戴尔',
    '戴尔（DELL)': 'Dell 戴尔',
    '机械革命': 'MECHREVO 机械革命',
    '机械革命（MECHREVO）': 'MECHREVO 机械革命',
    '雷神': 'ThundeRobot 雷神',
    '雷神(ThundeRobot)': 'ThundeRobot 雷神',
    '便携本': 'Laptop 便携本',
    '耳机': '其它耳机',
    '智能手表': '其它智能手表',
    '平板': 'Tablet 平板',
    'realme': 'Realme 真我',
    '真我': 'Realme 真我',
    '一加': 'OnePlus 一加',
    'NULL': 'Unknown'
}

# 定义函数来规范化品牌名称
def normalize_brand(brand):
    if pd.isna(brand):
        return 'Unknown'
    
    # 检查是否为Apple设备
    brand_str = str(brand)
    if any(device in brand_str for device in ['iPhone', 'MacBook', 'iPad', 'Apple', 'Watch', 'AirPods']):
        return 'Apple 苹果'
    
    # 检查品牌是否在映射字典中
    if brand in brand_mapping:
        return brand_mapping[brand]
    
    # 如果不在字典中，尝试部分匹配
    for key, value in brand_mapping.items():
        if key.lower() in brand.lower() or brand.lower() in key.lower():
            return value
    
    # 如果没有匹配到，返回原始品牌名
    return brand

# 定义函数来过滤非电子设备数据和营销文本
def clean_device_data(row):
    # 排除特定的非电子设备品牌
    excluded_brands = ['沪上阿姨']
    if row['pro_brand'] in excluded_brands:
        return False
    
    # 移除营销相关文本
    marketing_terms = ['极速发货', '长租推荐', '分期', '极速配送', '公开零售版', '标准版', '专属版']
    for term in marketing_terms:
        row['pro_title'] = row['pro_title'].replace(term, '').strip()
        row['pro_model'] = row['pro_model'].replace(term, '').strip()
    
    return True

# 应用品牌规范化
df['normalized_brand'] = df['pro_brand'].apply(normalize_brand)

# 识别设备类型
def identify_device_category(row):
    title = str(row['pro_title'])
    model = str(row['pro_model'])
    brand = row['normalized_brand']
    
    # 检查是否为笔记本电脑相关设备
    if re.search(r'Laptop|便携本|笔记本|Book|本|电脑', title, re.IGNORECASE):
        return 'Computer'
    
    # 只有手机分类的品牌
    smartphone_only_brands = ['魅族', 'Samsung 三星', 'Nubia 努比亚', 'Realme 真我']
    if brand in smartphone_only_brands:
        return 'Smartphone'
    
    # 只有电脑分类的品牌
    computer_only_brands = ['Lenovo 联想', 'HP 惠普', 'Dell 戴尔']
    if brand in computer_only_brands:
        return 'Computer'
    
    # Apple设备类型识别
    if brand == 'Apple':
        if re.search(r'MacBook|Mac\s*Pro|iMac', title, re.IGNORECASE):
            return 'Computer'
        elif re.search(r'iPhone', title, re.IGNORECASE):
            return 'Smartphone'
        elif re.search(r'iPad', title, re.IGNORECASE):
            return 'Tablet'
        elif re.search(r'Watch', title, re.IGNORECASE):
            return 'Smartwatch'
        elif re.search(r'AirPods|Pods', title, re.IGNORECASE):
            return 'Earphone'
    
    # 华为设备类型识别
    elif brand == 'Huawei 华为':
        if re.search(r'Mate\s*\d+|P\s*\d+|nova\s*\d+|畅享\s*\d+|Pura\s*\d+|Pocket\s*\d+', title, re.IGNORECASE):
            return 'Smartphone'
        elif re.search(r'MatePad|平板', title, re.IGNORECASE):
            return 'Tablet'
        elif re.search(r'MateBook|笔记本|Book|本|电脑', title, re.IGNORECASE):
            return 'Computer'
        elif re.search(r'WATCH|手表|Band', title, re.IGNORECASE):
            return 'Smartwatch'
        elif re.search(r'FreeBuds|耳机|Buds', title, re.IGNORECASE):
            return 'Earphone'
    
    # 华硕设备类型识别
    elif brand == 'Asus 华硕':
        if re.search(r'手机|Phone|ROG', title, re.IGNORECASE):
            return 'Smartphone'
        else:
            return 'Computer'
    
    # 荣耀设备类型识别
    elif brand == 'Honor 荣耀':
        if re.search(r'笔记本|Book|本|电脑', title, re.IGNORECASE):
            return 'Computer'
        else:
            return 'Smartphone'
    
    # 其他品牌默认为手机
    return 'Smartphone'

# 提取设备型号信息
def extract_model_info(row):
    title = str(row['pro_title'])
    model = str(row['pro_model'])
    brand = row['normalized_brand']
    
    # 首先尝试从title中提取完整的型号名称
    def extract_model_from_title(title):
        # 移除一些通用的营销文本
        marketing_terms = ['全新国行', '【95新】', '【99新】', '[非监管机]', '[不上征信]', '租满6期支持归还', 
                         '租物高分专享', '5G全网通', 'WiFi版', '蜂窝版']
        for term in marketing_terms:
            title = title.replace(term, '').strip()
        
        # 提取主要型号信息（第一个数字之前的部分加上紧跟的数字和后缀）
        model_match = re.search(r'([a-zA-Z\s]+(?:\d+)?(?:\s*(?:Pro|Plus|Max|Ultra|Air|mini|Pro\s*Max|ProMax))?)', title)
        if model_match:
            return model_match.group(1).strip()
        return None

    # 尝试从title提取型号
    model_name = extract_model_from_title(title)
    if model_name:
        return model_name

    # 如果从title无法提取到有效型号，再使用原有的提取逻辑
    def extract_generic_model(text, pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            model_num = match.group(1)
            model_type = match.group(2) if len(match.groups()) > 1 and match.group(2) else ''
            # 标准化Pro Max/ProMax表示
            if model_type and model_type.lower() == 'promax':
                model_type = 'Pro Max'
            return f"{match.group(0)}"
        return None
    
    # 针对Apple设备的特殊处理
    if brand == 'Apple':
        # 提取iPhone型号
        iphone_match = re.search(r'iPhone\s*(\d+)(?:\s*(Pro\s*Max|ProMax|Pro|Plus))?', title, re.IGNORECASE)
        if iphone_match:
            model_num = iphone_match.group(1)
            model_type = iphone_match.group(2) if iphone_match.group(2) else ''
            if model_type and model_type.lower() == 'promax':
                model_type = 'Pro Max'
            return f"iPhone {model_num}{' ' + model_type if model_type else ''}"
        
        # 提取iPad型号
        ipad_match = re.search(r'iPad(?:\s*(Pro|Air|mini))?\s*(\d+)?(?:\s*(\d+)?英寸)?', title, re.IGNORECASE)
        if ipad_match:
            model_type = ipad_match.group(1) if ipad_match.group(1) else ''
            model_num = ipad_match.group(2) if ipad_match.group(2) else ''
            return f"iPad{' ' + model_type if model_type else ''}{' ' + model_num if model_num else ''}"
        
        # 提取MacBook型号
        mac_match = re.search(r'MacBook\s*(Air|Pro)?\s*(\d+)?', title, re.IGNORECASE)
        if mac_match:
            model_type = mac_match.group(1) if mac_match.group(1) else ''
            return f"MacBook{' ' + model_type if model_type else ''}"
        
        # 提取Watch型号
        watch_match = re.search(r'(Apple\s*Watch|Watch)\s*(Ultra|Series)?\s*(\d+)?', title, re.IGNORECASE)
        if watch_match:
            model_type = watch_match.group(2) if watch_match.group(2) else ''
            model_num = watch_match.group(3) if watch_match.group(3) else ''
            return f"Apple Watch{' ' + model_type if model_type else ''}{' ' + model_num if model_num else ''}"
        
        # 提取AirPods型号
        airpods_match = re.search(r'AirPods\s*(Pro|Max)?\s*(\d+)?', title, re.IGNORECASE)
        if airpods_match:
            model_type = airpods_match.group(1) if airpods_match.group(1) else ''
            model_num = airpods_match.group(2) if airpods_match.group(2) else ''
            return f"AirPods{' ' + model_type if model_type else ''}{' ' + model_num if model_num else ''}"
    
    # 针对华为设备的特殊处理
    elif brand == 'Huawei 华为':
        # 华为Mate系列
        mate_match = re.search(r'Mate\s*(\d+[XSE]*|X|Xs|pad|Book)\s*(Pro\s*Max|ProMax|Pro|Plus|Pro\+|RS|青春版)?', title, re.IGNORECASE)
        if mate_match:
            model_num = mate_match.group(1)
            model_type = mate_match.group(2) if mate_match.group(2) else ''
            if model_type and model_type.lower() == 'promax':
                model_type = 'Pro Max'
            return f"Mate {model_num}{' ' + model_type if model_type else ''}"
        
        # 华为nova系列
        nova_match = re.search(r'[Nn]ova\s*(\d+[i]*|Play\d*|Smart)\s*(Pro|Plus|Flip|青春版)?', title)
        if nova_match:
            model_num = nova_match.group(1)
            model_type = nova_match.group(2) if nova_match.group(2) else ''
            return f"Nova {model_num}{' ' + model_type if model_type else ''}"
        
        # 华为Pura系列
        pura_match = re.search(r'Pura\s*(\d+)\s*(Pro|Plus|Ultra)?', title, re.IGNORECASE)
        if pura_match:
            model_num = pura_match.group(1)
            model_type = pura_match.group(2) if pura_match.group(2) else ''
            return f"Pura {model_num}{' ' + model_type if model_type else ''}"
        
        # 华为Pocket系列
        pocket_match = re.search(r'Pocket\s*(\d+|S)?', title, re.IGNORECASE)
        if pocket_match:
            model_num = pocket_match.group(1) if pocket_match.group(1) else ''
            return f"Pocket{' ' + model_num if model_num else ''}"
        
        # 华为MatePad系列
        matepad_match = re.search(r'MatePad\s*(Pro|Air)?\s*(\d+)?(?:\s*(\d+(?:\.\d+)?)?英寸)?', title, re.IGNORECASE)
        if matepad_match:
            model_type = matepad_match.group(1) if matepad_match.group(1) else ''
            model_num = matepad_match.group(2) if matepad_match.group(2) else ''
            screen_size = matepad_match.group(3) if matepad_match.group(3) else ''
            return f"MatePad{' ' + model_type if model_type else ''}{' ' + model_num if model_num else ''}{' ' + screen_size + '英寸' if screen_size else ''}"
        
        # 华为MateBook系列
        matebook_match = re.search(r'MateBook\s*([A-Z]\d+[XE]*|Air|Pro)\s*(\d+)?', title, re.IGNORECASE)
        if matebook_match:
            model_type = matebook_match.group(1) if matebook_match.group(1) else ''
            model_num = matebook_match.group(2) if matebook_match.group(2) else ''
            return f"MateBook {model_type}{' ' + model_num if model_num else ''}"
        
        # 华为Watch系列
        watch_match = re.search(r'WATCH\s*(GT|FIT)?\s*(\d+[e]*|Pro|Ultimate)?', title, re.IGNORECASE)
        if watch_match:
            model_type = watch_match.group(1) if watch_match.group(1) else ''
            model_num = watch_match.group(2) if watch_match.group(2) else ''
            return f"WATCH{' ' + model_type if model_type else ''}{' ' + model_num if model_num else ''}"
        
        # 华为FreeBuds系列
        freebuds_match = re.search(r'FreeBuds\s*(Pro|Studio)?\s*(\d+[e]*)?', title, re.IGNORECASE)
        if freebuds_match:
            model_type = freebuds_match.group(1) if freebuds_match.group(1) else ''
            model_num = freebuds_match.group(2) if freebuds_match.group(2) else ''
            return f"FreeBuds{' ' + model_type if model_type else ''}{' ' + model_num if model_num else ''}"
        
        # 华为P系列
        p_match = re.search(r'P\s*(\d+[E]*|Smart|Play)\s*(Pro|Plus|Lite|青春版)?', title)
        if p_match:
            model_num = p_match.group(1)
            model_type = p_match.group(2) if p_match.group(2) else ''
            return f"P {model_num}{' ' + model_type if model_type else ''}"
        
        # 华为畅享系列
        enjoy_match = re.search(r'畅享\s*(\d+[e]*|Plus|Pro|Z|Max)\s*(Pro|Plus|5G)?', title)
        if enjoy_match:
            model_num = enjoy_match.group(1)
            model_type = enjoy_match.group(2) if enjoy_match.group(2) else ''
            return f"畅享 {model_num}{' ' + model_type if model_type else ''}"
    
    # 针对小米设备的特殊处理
    elif brand == 'Xiaomi':
        # 小米数字系列
        mi_match = re.search(r'小米\s*(\d+[SXE]*|Note\s*\d+|Mix\s*\d*|Max\s*\d*)\s*(Pro|Plus|Ultra|青春版|S|T)?', title)
        if mi_match:
            model_num = mi_match.group(1)
            model_type = mi_match.group(2) if mi_match.group(2) else ''
            return f"小米 {model_num}{' ' + model_type if model_type else ''}"
        
        # Redmi系列
        redmi_match = re.search(r'[Rr]edmi\s*(\d+[AXCE]*|Note\s*\d+[AXCE]*|K\d+)\s*(Pro|Plus|Ultra|青春版|S|T)?', title)
        if redmi_match:
            model_num = redmi_match.group(1)
            model_type = redmi_match.group(2) if redmi_match.group(2) else ''
            return f"Redmi {model_num}{' ' + model_type if model_type else ''}"
    
    # 针对OPPO设备的特殊处理
    elif brand == 'OPPO':
        # Reno系列
        reno_match = re.search(r'[Rr]eno\s*(\d+[\+]*|Ace\d*|Z\d*)\s*(Pro|Plus|Ultra|青春版)?', title)
        if reno_match:
            model_num = reno_match.group(1)
            model_type = reno_match.group(2) if reno_match.group(2) else ''
            return f"Reno {model_num}{' ' + model_type if model_type else ''}"
        
        # Find系列
        find_match = re.search(r'[Ff]ind\s*(\d+|X\d*|N\d*)\s*(Pro|Plus|Ultra|青春版)?', title)
        if find_match:
            model_num = find_match.group(1)
            model_type = find_match.group(2) if find_match.group(2) else ''
            return f"Find {model_num}{' ' + model_type if model_type else ''}"
        
        # A系列
        a_match = re.search(r'OPPO\s*A\s*(\d+[sek]*|Ace)\s*(Pro|Plus|Ultra|青春版)?', title, re.IGNORECASE)
        if a_match:
            model_num = a_match.group(1)
            model_type = a_match.group(2) if a_match.group(2) else ''
            return f"A {model_num}{' ' + model_type if model_type else ''}"
    
    # 针对vivo设备的特殊处理
    elif brand == 'VIVO':
        # X系列
        x_match = re.search(r'[Xx](\d+[ste]*|Fold|Note)\s*(Pro|Plus|Ultra|青春版)?', title)
        if x_match:
            model_num = x_match.group(1)
            model_type = x_match.group(2) if x_match.group(2) else ''
            return f"X {model_num}{' ' + model_type if model_type else ''}"
        
        # Y系列
        y_match = re.search(r'[Yy](\d+[se]*|Play)\s*(Pro|Plus|Ultra|青春版)?', title)
        if y_match:
            model_num = y_match.group(1)
            model_type = y_match.group(2) if y_match.group(2) else ''
            return f"Y {model_num}{' ' + model_type if model_type else ''}"
        
        # iQOO系列
        iqoo_match = re.search(r'iQOO\s*(\d+[se]*|Neo\d*|Pro\d*)\s*(Pro|Plus|Ultra|青春版)?', title, re.IGNORECASE)
        if iqoo_match:
            model_num = iqoo_match.group(1)
            model_type = iqoo_match.group(2) if iqoo_match.group(2) else ''
            return f"iQOO {model_num}{' ' + model_type if model_type else ''}"
    
    # 针对三星设备的特殊处理
    elif brand == 'Samsung':
        # Galaxy S系列
        s_match = re.search(r'[Ss](\d+[\+e]*|Ultra)\s*(\+|Plus|Ultra|FE)?', title)
        if s_match:
            model_num = s_match.group(1)
            model_type = s_match.group(2) if s_match.group(2) else ''
            return f"Galaxy S{model_num}{' ' + model_type if model_type else ''}"
        
        # Galaxy Note系列
        note_match = re.search(r'[Nn]ote\s*(\d+[\+]*|Ultra)\s*(\+|Plus|Ultra|FE)?', title)
        if note_match:
            model_num = note_match.group(1)
            model_type = note_match.group(2) if note_match.group(2) else ''
            return f"Galaxy Note {model_num}{' ' + model_type if model_type else ''}"
        
        # Galaxy A系列
        a_match = re.search(r'[Aa](\d+[se]*|Quantum)\s*(s|e|5G)?', title)
        if a_match:
            model_num = a_match.group(1)
            model_type = a_match.group(2) if a_match.group(2) else ''
            return f"Galaxy A{model_num}{' ' + model_type if model_type else ''}"
        
        # Galaxy Z系列
        z_match = re.search(r'[Zz]\s*(Fold|Flip)\s*(\d+|5G)?', title)
        if z_match:
            model_type = z_match.group(1)
            model_num = z_match.group(2) if z_match.group(2) else ''
            return f"Galaxy {model_type}{' ' + model_num if model_num else ''}"
    
    # 提取内存配置信息
    memory_match = re.search(r'(\d+)[Gg][Bb]?[\+\s]*(\d+)[Gg][Bb]', model)
    if memory_match:
        ram = memory_match.group(1)
        storage = memory_match.group(2)
        return f"{ram}GB+{storage}GB"
    
    # 如果没有特殊处理，尝试从model字段提取信息
    if '+' in model:
        # 可能包含内存信息，如"12G+256G"
        return model.split('+')[0]
    return model

# 应用设备类型识别和型号提取
df['device_category'] = df.apply(identify_device_category, axis=1)
df['normalized_model'] = df.apply(extract_model_info, axis=1)

# 提取内存和存储容量信息
def extract_memory_storage(row):
    model = str(row['pro_model'])
    title = str(row['pro_title'])
    
    # 尝试匹配常见的内存+存储格式，如"8GB+128GB"、"12G+256G"等
    memory_storage_match = re.search(r'(\d+)\s*[Gg][Bb]?\s*[+\s]\s*(\d+)\s*[Gg][Bb]?', model + ' ' + title)
    if memory_storage_match:
        ram = memory_storage_match.group(1)
        storage = memory_storage_match.group(2)
        return f"{ram}GB+{storage}GB"
    
    # 尝试单独匹配存储容量
    storage_match = re.search(r'(\d+)\s*([GT][B]?)', model + ' ' + title, re.IGNORECASE)
    if storage_match:
        size = storage_match.group(1)
        unit = storage_match.group(2).upper()
        if unit in ['G', 'GB']:
            return f"未知+{size}GB"
        elif unit in ['T', 'TB']:
            size_gb = int(size) * 1024
            return f"未知+{size_gb}GB"
    
    return "未知"

# 提取颜色信息
def extract_color(row):
    color = str(row['pro_color'])
    title = str(row['pro_title'])
    
    # 标准化颜色映射字典
    color_mapping = {
        '黑色': ['黑', '曜石黑', '深空黑', '幻夜黑', '玄黑', '碳晶黑'],
        '白色': ['白', '珍珠白', '冰霜白', '晨曦白', '釉白'],
        '蓝色': ['蓝', '海蓝', '天空蓝', '极光蓝', '渐变蓝'],
        '绿色': ['绿', '翡翠绿', '青山绿', '薄荷绿'],
        '紫色': ['紫', '梦幻紫', '星云紫', '幻彩紫'],
        '金色': ['金', '流沙金', '香槟金', '玫瑰金'],
        '银色': ['银', '星光银', '钛银', '月光银'],
        '灰色': ['灰', '深空灰', '曜石灰', '星空灰'],
        '红色': ['红', '中国红', '珊瑚红', '热力红']
    }
    
    # 尝试从颜色字段匹配标准颜色
    for standard_color, variants in color_mapping.items():
        if any(variant in color for variant in variants):
            return standard_color
    
    # 尝试从标题中提取颜色信息
    for standard_color, variants in color_mapping.items():
        if any(variant in title for variant in variants):
            return standard_color
    
    # 如果找不到匹配的标准颜色，返回原始颜色
    return color if color and color != 'nan' else '未知'

# 应用内存和存储容量提取
df['memory_storage'] = df.apply(extract_memory_storage, axis=1)

# 应用颜色提取
df['normalized_color'] = df.apply(extract_color, axis=1)

# 应用存储容量提取
df['storage'] = df.apply(extract_memory_storage, axis=1)

# 保存处理后的数据
df.to_csv('/Users/chenyonglin/myCode/gitee/myWork/Python/Equipments/normalized_data.csv', index=False)

# 创建Excel写入器
with pd.ExcelWriter('/Users/chenyonglin/myCode/gitee/myWork/Python/Equipments/equipment_stats.xlsx') as writer:
    # 1. 原始品牌和型号统计
    original_brand_model_stats = df.groupby(['pro_brand', 'pro_model']).agg({
        '数量': 'sum'
    }).reset_index()
    original_brand_model_stats = original_brand_model_stats.sort_values(['pro_brand', 'pro_model'])
    original_brand_model_stats.to_excel(writer, sheet_name='原始品牌型号统计', index=False)
    
    # 2. 规范化品牌和设备类型统计
    brand_category_stats = df.groupby(['normalized_brand', 'device_category']).agg({
        '数量': 'sum'
    }).reset_index()
    brand_category_stats = brand_category_stats.sort_values(['normalized_brand', 'device_category'])
    brand_category_stats.to_excel(writer, sheet_name='品牌设备类型统计', index=False)
    
    # 3. 规范化品牌、设备类型和型号统计
    brand_category_model_stats = df.groupby(['normalized_brand', 'device_category', 'normalized_model']).agg({
        '数量': 'sum'
    }).reset_index()
    brand_category_model_stats = brand_category_model_stats.sort_values(['normalized_brand', 'device_category', 'normalized_model'])
    brand_category_model_stats.to_excel(writer, sheet_name='品牌设备类型型号统计', index=False)
    
    # 4. 品牌、设备类型和存储容量统计
    storage_stats = df.groupby(['normalized_brand', 'device_category', 'storage']).agg({
        '数量': 'sum'
    }).reset_index()
    storage_stats = storage_stats.sort_values(['normalized_brand', 'device_category', 'storage'])
    storage_stats.to_excel(writer, sheet_name='存储容量统计', index=False)
    
    # 4. 品牌和颜色统计
    color_stats = df.groupby(['normalized_brand', 'pro_color']).agg({
        '数量': 'sum'
    }).reset_index()
    color_stats = color_stats.sort_values(['normalized_brand', 'pro_color'])
    color_stats.to_excel(writer, sheet_name='颜色统计', index=False)
    
    # 5. 品牌总量统计（原始品牌）
    original_brand_total = df.groupby('pro_brand').agg({
        '数量': 'sum'
    }).reset_index()
    original_brand_total = original_brand_total.sort_values('数量', ascending=False)
    original_brand_total.to_excel(writer, sheet_name='原始品牌总量', index=False)
    
    # 6. 品牌总量统计（规范化品牌）
    brand_total = df.groupby('normalized_brand').agg({
        '数量': 'sum'
    }).reset_index()
    brand_total = brand_total.sort_values('数量', ascending=False)
    brand_total.to_excel(writer, sheet_name='规范化品牌总量', index=False)
    
    # 7. 存储容量分布统计
    storage_dist = df.groupby('storage').agg({
        '数量': 'sum'
    }).reset_index()
    storage_dist = storage_dist.sort_values('数量', ascending=False)
    storage_dist.to_excel(writer, sheet_name='存储容量分布', index=False)

print("数据统计完成！统计结果已保存到equipment_stats.xlsx文件中。\n")

# 创建交互式命令行界面
def display_brands():
    # 获取品牌总量统计
    brand_total = df.groupby('normalized_brand').agg({
        '数量': 'sum'
    }).reset_index().sort_values('数量', ascending=False)
    
    print("=== 品牌设备总览 ===")
    brand_dict = {}
    for idx, (brand, count) in enumerate(zip(brand_total['normalized_brand'], brand_total['数量']), 1):
        print(f"{idx}. {brand}（{count} 台）")
        brand_dict[str(idx)] = brand
    return brand_dict

def display_device_categories(brand):
    # 获取指定品牌的设备类型统计
    if brand == 'Meizu 魅族':
        # 魅族品牌只显示手机类型
        category_stats = pd.DataFrame({
            'device_category': ['Smartphone'],
            '数量': [df[df['normalized_brand'] == brand]['数量'].sum()]
        })
    else:
        category_stats = df[df['normalized_brand'] == brand].groupby('device_category').agg({
            '数量': 'sum'
        }).reset_index().sort_values('数量', ascending=False)
    
    print(f"\n=== {brand} 设备类型分布 ===")
    category_dict = {}
    for idx, (category, count) in enumerate(zip(category_stats['device_category'], category_stats['数量']), 1):
        chinese_category = device_type_mapping.get(category, category)
        print(f"{idx}. {chinese_category}（{count} 台）")
        category_dict[str(idx)] = category
    
    return category_dict

# 创建颜色映射字典，合并相似颜色
color_mapping = {
    '黑色': '黑色钛金属',
    '黑色钛金属': '黑色钛金属',
    '白色': '白色钛金属',
    '白色钛金属': '白色钛金属',
    '蓝色': '蓝色钛金属',
    '蓝色钛金属': '蓝色钛金属',
    '自然色': '原色钛金属',
    '原色钛金属': '原色钛金属'
}

def normalize_color(color):
    return color_mapping.get(color, color)

def display_models(brand, category):
    # 获取指定品牌和设备类型的型号统计
    model_stats = df[(df['normalized_brand'] == brand) & 
                    (df['device_category'] == category)].groupby('normalized_model').agg({
        '数量': 'sum'
    }).reset_index()
    
    # 如果是Apple品牌，使用自定义排序逻辑
    if brand == 'Apple':
        # 创建一个函数来提取iPhone型号的数字和后缀
        def extract_iphone_info(model_name):
            # 默认值
            model_num = 0
            suffix_priority = 0
            
            # 提取iPhone型号信息
            iphone_match = re.search(r'iPhone\s*(\d+)(?:\s*(Pro\s*Max|ProMax|Pro|Plus))?', model_name, re.IGNORECASE)
            if iphone_match:
                model_num = int(iphone_match.group(1))
                suffix = iphone_match.group(2).lower() if iphone_match.group(2) else ''
                
                # 设置后缀优先级：Pro Max/ProMax > Pro > Plus > 无后缀
                if 'pro max' in suffix or 'promax' in suffix:
                    suffix_priority = 3
                elif 'pro' in suffix:
                    suffix_priority = 2
                elif 'plus' in suffix:
                    suffix_priority = 1
                else:
                    suffix_priority = 0
            
            return (model_num, suffix_priority)
        
        # 添加排序键
        model_stats['sort_key'] = model_stats['normalized_model'].apply(lambda x: extract_iphone_info(x) if 'iPhone' in x else (0, 0))
        
        # 按照自定义排序逻辑排序：先按型号数字降序，再按后缀优先级降序，最后按数量降序
        model_stats = model_stats.sort_values(
            by=['sort_key', '数量'], 
            ascending=[False, False],
            key=lambda x: x if x.name != 'sort_key' else pd.Series([tuple(i) for i in x])
        )
    else:
        # 非Apple品牌，按数量降序排序
        model_stats = model_stats.sort_values('数量', ascending=False)
    
    print(f"\n=== {brand} {category} 型号分布 ===")
    model_dict = {}
    for idx, row in enumerate(model_stats.iterrows(), 1):
        print(f"{idx}. {row[1]['normalized_model']}（{int(row[1]['数量'])} 台）")
        model_dict[str(idx)] = row[1]['normalized_model']
    
    return model_dict

def display_model_details(brand, category, model):
    # 获取指定型号的详细信息
    model_data = df[(df['normalized_brand'] == brand) & 
                   (df['device_category'] == category) & 
                   (df['normalized_model'] == model)].copy()
    
    # 规范化颜色
    model_data['normalized_color'] = model_data['pro_color'].apply(normalize_color)
    
    print(f"\n=== {model} 详细信息 ===")
    
    # 显示原始数据名称
    print("\n原始数据名称：")
    for title in model_data['pro_title'].unique():
        count = model_data[model_data['pro_title'] == title]['数量'].sum()
        print(f"- {title}（{int(count)} 台）")
    
    # 颜色分布统计
    color_stats = model_data.groupby('normalized_color').agg({
        '数量': 'sum'
    }).reset_index().sort_values('数量', ascending=False)
    
    print("\n颜色分布：")
    for _, row in color_stats.iterrows():
        print(f"{row['normalized_color']}（{int(row['数量'])} 台）")
    
    # 存储容量分布统计
    storage_stats = model_data.groupby('storage').agg({
        '数量': 'sum'
    }).reset_index().sort_values('数量', ascending=False)
    
    print("\n存储容量分布：")
    for _, row in storage_stats.iterrows():
        print(f"{row['storage']}（{int(row['数量'])} 台）")

# 主交互循环
while True:
    # 显示品牌列表和编号
    brand_dict = display_brands()
    
    # 获取用户输入的品牌编号
    brand_choice = input("\n请输入品牌编号查看详细信息（输入q退出）：")
    
    if brand_choice.lower() == 'q':
        print("\n感谢使用！再见！")
        break
    
    if brand_choice in brand_dict:
        while True:
            # 显示该品牌的设备类型
            category_dict = display_device_categories(brand_dict[brand_choice])
            
            # 获取用户输入的设备类型编号
            category_choice = input("\n请输入设备类型编号查看详细信息（直接回车返回上级菜单）：")
            
            if not category_choice:
                break
            
            if category_choice in category_dict:
                while True:
                    # 显示该品牌和设备类型的型号
                    model_dict = display_models(brand_dict[brand_choice], category_dict[category_choice])
                    
                    # 获取用户输入的型号编号
                    model_choice = input("\n请输入型号编号查看详细信息（直接回车返回上级菜单）：")
                    
                    if not model_choice:
                        break
                    
                    if model_choice in model_dict:
                        # 显示型号详细信息
                        display_model_details(brand_dict[brand_choice], category_dict[category_choice], model_dict[model_choice])
                        input("\n按回车键继续...")
                    else:
                        print("无效的型号编号，请重新输入")
            else:
                print("无效的设备类型编号，请重新输入")
    else:
        print("\n无效的品牌编号，请重新输入。")