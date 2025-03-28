import pandas as pd
import re

# 1. 手动建立机型库
# 根据CSV文件中的数据自动生成的机型库
device_library = [
    # Apple iPhone 系列
    "iPhone 16 Pro Max", "iPhone 16 Pro", "iPhone 16 Plus", "iPhone 16", "iPhone 16e",
    "iPhone 15 Pro Max", "iPhone 15 Pro", "iPhone 15 Plus", "iPhone 15",
    "iPhone 14 Pro Max", "iPhone 14 Pro", "iPhone 14 Plus", "iPhone 14",
    "iPhone 13 Pro Max", "iPhone 13 Pro", "iPhone 13", "iPhone 13 mini",
    "iPhone 12 Pro Max", "iPhone 12 Pro", "iPhone 12", "iPhone 12 mini",
    "iPhone 11 Pro Max", "iPhone 11 Pro", "iPhone 11",
    "iPhone X", "iPhone XS", "iPhone XS Max", "iPhone XR",
    
    # Apple Mac 系列
    "MacBook Air M3 2024款", "MacBook Air M2 2022款", 
    "MacBook Pro 2023款", "MacBook Pro 2022款",
    
    # Apple iPad 系列
    "iPad Pro 2017款", "iPad Pro 2018款", "iPad Pro 2019款", "iPad Pro 2020款", "iPad Pro 2021款", "iPad Pro 2022款", "iPad Pro 2023款", "iPad Pro 2024款", "iPad Pro 2025款", 
    "iPad Air 2017款", "iPad Air 2018款", "iPad Air 2019款", "iPad Air 2020款", "iPad Air 2021款", "iPad Air 2022款", "iPad Air 2023款", "iPad Air 2024款", "iPad Air 2025款", 
    "iPad 2017款", "iPad 2018款", "iPad 2019款", "iPad 2020款", "iPad 2021款", "iPad 2022款", "iPad 2023款", "iPad 2024款", "iPad 2025款", 
    "iPad mini 3", "iPad mini 4", "iPad mini 5", "iPad mini 6 2021款",
    
    # Apple Watch 系列
    "Apple Watch Ultra 1", "Apple Watch Ultra 2", "Apple Watch Ultra 3", "Apple Watch Ultra 4", "Apple Watch Ultra 5",
    "Apple Watch Series 1", "Apple Watch Series 2", "Apple Watch Series 3", "Apple Watch Series 4", "Apple Watch Series 5", 
    "Apple Watch Series 6", "Apple Watch Series 7", "Apple Watch Series 8", "Apple Watch Series 9", "Apple Watch Series 10", 
    "Apple Watch Series 11", "Apple Watch Series 12", 
    "Apple Watch SE 2020款", "Apple Watch SE 2021款", "Apple Watch SE 2022款", "Apple Watch SE 2023款", "Apple Watch SE 2024款",
    
    # Apple 配件
    # AirPods 系列 - 通用匹配任意代数
    "AirPods 1", "AirPods 2", "AirPods 3", "AirPods 4", "AirPods 5", "AirPods 6",
    "AirPods Pro 第一代", "AirPods Pro 第二代", "AirPods Pro 第三代", "AirPods Pro 第四代",
    "AirPods Max 第一代", "AirPods Max 第二代", "AirPods Max 第三代",
    # 华为系列
    "Huawei Mate 60 Pro", "华为 Mate 60 Pro+", "华为 Mate 60", "华为 Mate X5",
    "华为 Pura 70 Ultra", "华为 Pura 70 Pro+", "华为 Pura 70 Pro", "华为 Pura 70",
    "华为 nova 12 活力版", "华为 畅享70",
    "华为 MatePad Pro 11英寸 2022",
    "华为 MateBook D16 SE 2024", "华为 MateBook D16 2024", 
    "华为 MateBook D14 2024", "华为 MateBook 14 2024款", "华为 MateBook E 2023",
    
    # 华为配件
    "HUAWEI WATCH 4 Pro", "HUAWEI WATCH GT 4", "HUAWEI WATCH GT 3 Pro",
    "华为 WATCH Ultimate", "华为 WATCH 4 Pro", "华为 WATCH GT 4", "华为 Watch 4",
    "华为 FreeBuds Pro 3", "华为 FreeBuds Lipstick 2", "HUAWEI WATCH Buds",
    
    # 小米系列
    "Xiaomi 14 Ultra", "Xiaomi 14 Pro", "Xiaomi 14", "小米 14 Ultra", "小米 14", "小米 13",
    "小米 Civi 4 Pro", "小米 Civi 4 Pro",
    "小米 MIX Fold4", "小米 MIX Flip",
    "小米 Pad 6S Pro 12.4",
    
    # 红米系列
    "Redmi K70 Pro", "Redmi K70", "Redmi K70E", "红米 K70 至尊版", "红米 Turbo 3",
    
    # vivo系列
    "vivo X100 Ultra", "vivo X100s Pro", "vivo X100s", "vivo X100 Pro", "vivo X100",
    "vivo X Fold3 Pro", "vivo S19 Pro", "vivo S19", "vivo S18e",
    "vivo iQOO 12 Pro", "vivo iQOO Neo9S Pro+", "vivo iQOO Neo 9 Pro", "vivo iQOO Neo 9",
    "vivo iQOO Z9 Turbo", "vivo Pad 3 Pro",
    
    # OPPO系列
    "OPPO Find X7 Ultra", "OPPO Find X7", "OPPO Find X6",
    "OPPO Reno 12 Pro", "OPPO Reno 12", "OPPO K12", "OPPO A3 Pro", "OPPO Watch 4 Pro",
    
    # 三星系列
    "Samsung Galaxy S24 Ultra", "三星 Galaxy S24 Ultra",
    "三星 Galaxy Z Fold 6", "三星 Galaxy Z Fold 4", 
    "三星 Galaxy Z Flip 6", "三星 Galaxy Z Flip 4",
    
    # 荣耀系列
    "荣耀Magic V3", "荣耀 Magic6 Pro", "荣耀200 Pro", "荣耀90 Pro",
    "荣耀X50 GT", "荣耀X50",
    
    # 努比亚系列
    "红魔9S Pro+", "红魔9S Pro", "红魔9Pro",
    
    # 一加系列
    "一加 Ace 3 Pro", "一加 Ace 3V", "一加 Ace 3", "一加平板 Pro",
    
    # realme系列
    "realme GT6",
    
    # 笔记本电脑
    "华硕 天选 5 Pro", "华硕 天选 5", "华硕 天选 4", "华硕 天选 Air 2024",
    "华硕 ROG 枪神 8", "华硕 无畏 Pro15 2024", "华硕 灵耀 14 2024",
    "联想 拯救者 Y9000P 2024", "联想 拯救者 Y9000X 2024", "联想 拯救者 Y7000P 2024",
    "联想 拯救者 R9000P 2024", "联想 拯救者 R7000 2023款", "联想 拯救者 Y9000P 2023",
    "联想 小新笔记本 Pro 16 2024款", "联想 小新笔记本 Pro 14 2024款", "联想 小新笔记本 16 2024款", "联想 小新笔记本 16 2023锐龙版",
    "联想 小新 Pad Pro 2025 舒视版", "联想 小新 Pad 2024款",
    "ThinkBook 16+ 2024", "ThinkBook 16 2023", "ThinkBook 15 2023",
    "惠普 星Book Pro 14 2024",
    "机械革命 蛟龙 16 Pro",
    "雷神 911MT 2023 天行者", "雷神 猎刃 15 2024"
]

# 将机型库转换为小写并去除空格的版本，便于匹配
device_library_normalized = {model.lower().replace(" ", ""): model for model in device_library}

# 2. 读取 CSV 文件
# 使用实际文件路径
df = pd.read_csv('/Users/chenyonglin/myCode/gitee/myWork/Python/Equipments/phones.csv')


# 3. 清洗 pro_title 列
def clean_title(title):
    """清洗标题，去除无关词缀"""
    # 去除方括号及其内容
    title = re.sub(r'\[.*?\]', '', title)
    # 去除小括号及其内容
    title = re.sub(r'（.*?）|\(.*?\)', '', title)
    # 去除"-"后面的内容
    title = re.split(r'-', title)[0]
    # 去除【】及其内容
    title = re.sub(r'【.*?】', '', title)
    # 去除常见无用前缀和修饰词
    title = title.replace('全新国行', '').strip()
    title = title.replace('第三代骁龙7S', '').strip()
    title = title.replace('轻薄笔记本商务办公本', '').strip()

    # 统一华为品牌名称
    title = re.sub(r'(华为|HUAWEI|Huawei)\s*', '华为 ', title)
    
    # 统一小米品牌名称
    title = re.sub(r'(小米|Xiaomi|MI)\s*', '小米 ', title)
    
    # 统一三星品牌名称
    title = re.sub(r'(三星|Samsung)\s*', '三星 ', title)
    
    # 统一年份后缀格式
    title = re.sub(r'(\d{4})(?!\s*款)', r'\1款', title)
    
    # 统一 Apple Watch 系列命名
    # 处理 Series 系列
    title = re.sub(r'Apple\s*Watch\s*S(?:eries)?\s*(\d+)', r'Apple Watch Series \1', title)
    # 处理 Ultra 系列
    title = re.sub(r'Apple\s*Watch\s*Ultra\s*(\d+)', r'Apple Watch Ultra \1', title)
    # 处理 SE 系列（保持年份格式）
    title = re.sub(r'Apple\s*Watch\s*SE\s*(\d{4})', r'Apple Watch SE \1款', title)  # 未匹配：3050款/3060款电竞台式机（30天起租）
    
    title = title.replace('5G全网通', '').replace('首月1元', '').replace('芝麻推广租物', '').replace('租物高分专享', '').replace('支持主动降噪', '').replace('通过率高', '').replace('顺丰发货', '').replace('顺丰包邮', '').replace('橡胶表带', '').replace('智能手表', '')
    title = title.replace('仅激活', '').replace('全新正品', '').replace('国行正品', '').replace('非监管机', '').strip()
    
    # 确保品牌名称前后有空格
    title = title.replace('华为', ' 华为 ').replace('小米', ' 小米 ').replace('三星', ' 三星 ').replace('  ', ' ').strip()
    return title


# 应用清洗函数
df['cleaned_title'] = df['pro_title'].apply(clean_title)


# 4. 匹配机型库
def standardize_suffix(text):
    """统一设备名称中的后缀格式"""
    text = re.sub(r'pro\+', 'Pro+', text, flags=re.IGNORECASE)
    text = re.sub(r'pro\b', 'Pro', text, flags=re.IGNORECASE)
    text = re.sub(r'ultra\b', 'Ultra', text, flags=re.IGNORECASE)
    return text

def match_device(title):
    """将清洗后的标题与机型库匹配"""
    # 如果传入的标题为空，直接返回"未匹配"
    if not title or pd.isna(title):
        return "未匹配: 空标题"
    
    # 规范化标题（小写并去除空格）
    normalized_title = title.lower().replace(" ", "")
    
    # 统一品牌名称处理
    normalized_title = normalized_title.replace("huawei", "华为").replace("huaweihuawe", "华为")
    normalized_title = normalized_title.replace("xiaomi", "小米").replace("xiaomixiaomi", "小米")
    normalized_title = normalized_title.replace("samsung", "三星").replace("三星三星", "三星")
    
    # 直接检查标题中是否包含关键词"airpods"
    if "airpods" in normalized_title.lower():
        # 处理AirPods系列命名标准化 
        airpods_pattern = r'airpods\s*(pro|max)?\s*(第)?(一|二|三|四|五|1|2|3|4|5)?(代)?'
        airpods_match = re.search(airpods_pattern, normalized_title, re.IGNORECASE)
        if airpods_match:
            model_type = airpods_match.group(1) or ""
            has_di = airpods_match.group(2) is not None
            generation = airpods_match.group(3) or "一"
            has_dai = airpods_match.group(4) is not None
            
            # 统一数字转为中文数字
            if generation in ['1', '一']:
                gen_cn = "一"
            elif generation in ['2', '二']:
                gen_cn = "二"
            elif generation in ['3', '三']:
                gen_cn = "三"
            elif generation in ['4', '四']:
                gen_cn = "四"
            elif generation in ['5', '五']:
                gen_cn = "五"
            else:
                gen_cn = generation
                
            # 构建标准格式
            if model_type and model_type.lower() == "pro":
                return f"AirPods Pro 第{gen_cn}代"
            elif model_type and model_type.lower() == "max":
                return f"AirPods Max 第{gen_cn}代"
            else:
                return f"AirPods {generation}"
    
    # 处理Apple Watch系列
    if "applewatch" in normalized_title.lower() or "apple watch" in title.lower():
        # 处理Series系列
        series_pattern = r'(?:apple\s*watch\s*)?series\s*(\d+)'
        series_match = re.search(series_pattern, normalized_title, re.IGNORECASE)
        if series_match:
            series_num = series_match.group(1)
            return f"Apple Watch Series {series_num}"
        
        # 处理Ultra系列
        ultra_pattern = r'(?:apple\s*watch\s*)?ultra\s*(\d+)'
        ultra_match = re.search(ultra_pattern, normalized_title, re.IGNORECASE)
        if ultra_match:
            ultra_num = ultra_match.group(1)
            return f"Apple Watch Ultra {ultra_num}"
        
        # 处理SE系列
        se_pattern = r'(?:apple\s*watch\s*)?se\s*(\d{4})?款?'
        se_match = re.search(se_pattern, normalized_title, re.IGNORECASE)
        if se_match:
            year = se_match.group(1)
            if year:
                return f"Apple Watch SE {year}款"
            else:
                return "Apple Watch SE"
    
    # 处理红米系列命名标准化
    redmi_pattern = r'redmi\s*(.+)'
    redmi_match = re.search(redmi_pattern, normalized_title, re.IGNORECASE)
    if redmi_match:
        model = redmi_match.group(1)
        # 统一Pro和Ultra的大小写格式
        model = re.sub(r'pro\+', 'Pro+', model, flags=re.IGNORECASE)
        model = re.sub(r'pro\b', 'Pro', model, flags=re.IGNORECASE)
        model = re.sub(r'ultra\b', 'Ultra', model, flags=re.IGNORECASE)
        # 将单个英文字母转换为大写
        model = re.sub(r'\b[a-z]\b', lambda x: x.group().upper(), model)
        return f"红米 {model}"
    
    # 处理ROG系列命名标准化
    rog_pattern = r'rog(游戏手机|魔霸)(\d+)\s*((?:plus|pro|ultra))?'
    rog_match = re.search(rog_pattern, normalized_title, re.IGNORECASE)
    if rog_match:
        device_type = rog_match.group(1)
        number = rog_match.group(2)
        suffix = rog_match.group(3)
        if suffix:
            return f"ROG {device_type} {number} {standardize_suffix(suffix)}"
        return f"ROG {device_type} {number}"
    
    # 处理ThinkBook系列命名标准化
    thinkbook_pattern = r'thinkbook\s*(.+)'
    thinkbook_match = re.search(thinkbook_pattern, normalized_title, re.IGNORECASE)
    if thinkbook_match:
        model = thinkbook_match.group(1)
        # 统一Pro和Ultra的大小写格式
        model = re.sub(r'pro\+', 'Pro+', model, flags=re.IGNORECASE)
        model = re.sub(r'pro\b', 'Pro', model, flags=re.IGNORECASE)
        model = re.sub(r'ultra\b', 'Ultra', model, flags=re.IGNORECASE)
        return f"ThinkBook {model}"
    
    # 处理Apple品牌设备
    normalized_title = re.sub(r'(apple|苹果)\s*(iphone|ipad|watch|airpods)', r'\2', normalized_title, flags=re.IGNORECASE)
    
    # 处理OPPO Reno系列命名标准化
    reno_pattern = r'oppo\s*reno\s*(\d+)\s*((?:pro|ultra))?'
    reno_match = re.search(reno_pattern, normalized_title, re.IGNORECASE)
    if reno_match:
        number = reno_match.group(1)
        suffix = reno_match.group(2)
        if suffix:
            return f"OPPO Reno {number} {standardize_suffix(suffix)}"
        return f"OPPO Reno {number}"
    
    # 处理华为Pura系列命名标准化
    pura_pattern = r'(?:华为\s*)?pura\s*(\d+)\s*((?:pro|ultra|pro\+))?'
    pura_match = re.search(pura_pattern, normalized_title, re.IGNORECASE)
    if pura_match:
        number = pura_match.group(1)
        suffix = pura_match.group(2)
        if suffix:
            return f"华为 Pura {number} {standardize_suffix(suffix)}"
        return f"华为 Pura {number}"
    
    # 统一年份后缀格式
    normalized_title = re.sub(r'(\d{4})(?!\s*款)', r'\1款', normalized_title)
    # 处理连续年份，添加斜杠分隔符
    normalized_title = re.sub(r'(\d{2})款(\d{2})款', r'\1款/\2款', normalized_title)
    
    # 精确匹配
    if normalized_title in device_library_normalized:
        return device_library_normalized[normalized_title]
    
    # 部分匹配：检查机型库中的每个标准名称是否包含在标题中
    for norm_model, standard_model in device_library_normalized.items():
        # 1. 直接匹配规范化的标题和机型名称
        if norm_model in normalized_title:
            return standard_model
        
        # 2. 对于iPhone，实施精确匹配（避免"iPhone 11"匹配成"iPhone 1"）
        if "iphone" in norm_model and "iphone" in normalized_title:
            # 提取iPhone型号
            iphone_pattern = r'iphone\s*(\d+\s*(?:plus|pro|promax|mini)?)'
            title_match = re.search(iphone_pattern, normalized_title, re.IGNORECASE)
            model_match = re.search(iphone_pattern, norm_model, re.IGNORECASE)
            
            if title_match and model_match and model_match.group(1) == title_match.group(1):
                return standard_model
        
        # 3. 对于iPad，实施更灵活的匹配（包括年份及型号）
        elif "ipad" in norm_model and "ipad" in normalized_title:
            # 提取iPad型号和年份
            ipad_pattern = r'ipad\s*(pro|air|mini)?\s*(\d+)?(?:[^年代数]*)(\d{4})?款?'
            title_match = re.search(ipad_pattern, normalized_title, re.IGNORECASE)
            model_match = re.search(ipad_pattern, norm_model, re.IGNORECASE)
            
            if title_match and model_match:
                # 检查是否匹配了相同的iPad型号（Pro/Air/mini）
                title_type = title_match.group(1) or ""
                model_type = model_match.group(1) or ""
                
                if title_type == model_type:
                    # 检查尺寸是否匹配（如有）
                    title_size = title_match.group(2) or ""
                    model_size = model_match.group(2) or ""
                    
                    if title_size == model_size or not title_size or not model_size:
                        # 检查年份是否匹配（如有）
                        title_year = title_match.group(3) or ""
                        model_year = model_match.group(3) or ""
                        
                        if title_year == model_year or not title_year or not model_year:
                            return standard_model
        
        # 对华为产品进行特殊处理
        elif "华为" in norm_model and "华为" in normalized_title:
            # 提取华为后面的型号部分进行匹配
            huawei_model_pattern = r'华为(.+)'
            title_match = re.search(huawei_model_pattern, normalized_title)
            model_match = re.search(huawei_model_pattern, norm_model)
            
            if title_match and model_match and model_match.group(1) in title_match.group(1):
                return standard_model
        
        # 对小米产品进行特殊处理
        elif "小米" in norm_model and "小米" in normalized_title:
            # 提取小米后面的型号部分进行匹配
            xiaomi_model_pattern = r'小米(.+)'
            title_match = re.search(xiaomi_model_pattern, normalized_title)
            model_match = re.search(xiaomi_model_pattern, norm_model)
            
            if title_match and model_match and model_match.group(1) in title_match.group(1):
                return standard_model
        
        # 对三星产品进行特殊处理
        elif "三星" in norm_model and "三星" in normalized_title:
            # 提取三星后面的型号部分进行匹配
            samsung_model_pattern = r'三星(.+)'
            title_match = re.search(samsung_model_pattern, normalized_title)
            model_match = re.search(samsung_model_pattern, norm_model)
            
            if title_match and model_match and model_match.group(1) in title_match.group(1):
                return standard_model

        # 以下是联想笔记本、小新系列等其他处理逻辑...
        # ... 略过一些代码以保持简洁 ...
        
    # 如果没有找到匹配，返回未匹配信息
    return f"未匹配: {title}"


# 应用匹配函数
df['standard_model'] = df['cleaned_title'].apply(match_device)

# 添加后处理步骤，专门处理小新笔记本的型号格式
def postprocess_xiaoxin_model(model_name):
    if '小新笔记本' in model_name:
        # 处理"1420款/23款"格式 => "14 20款/23款"
        model_name = re.sub(r'小新笔记本\s+(1[46])(\d{2})款/(\d{2})款', r'小新笔记本 \1 \2款/\3款', model_name)
        # 处理"1620款/23款"格式 => "16 20款/23款"
        model_name = re.sub(r'小新笔记本\s+1620款', r'小新笔记本 16 20款', model_name)
        # 处理"1420款"格式 => "14 20款"
        model_name = re.sub(r'小新笔记本\s+1420款', r'小新笔记本 14 20款', model_name)
        
        # 处理"小新笔记本 14轻薄笔记本商务办公本"的情况，去除多余后缀
        model_name = re.sub(r'小新笔记本\s+14(?:轻薄笔记本商务办公本|轻薄本|商务办公本|商务本|办公本).*', r'小新笔记本 14', model_name)
        
        # 处理类似情况，如果14后面不是数字也不是Pro等常规型号标识，就只保留"小新笔记本 14"
        if re.search(r'小新笔记本\s+14(?![\d\s]|Pro|轻)', model_name):
            model_name = re.sub(r'(小新笔记本\s+14).*', r'\1', model_name)
    
    return model_name

# 添加通用后处理函数，处理所有机型的格式规范化
def format_model_name(model_name):
    # 1. 将"WATCH"改为"Watch"
    model_name = re.sub(r'\bWATCH\b', 'Watch', model_name)
    
    # 2. 处理iPad机型格式，确保年份在机型名称的最后
    # 处理"2024款 iPad Pro M4芯片" => "iPad Pro 2024款"
    model_name = re.sub(r'(\d{4})款\s+(iPad\s+(?:Pro|Air|mini)(?:\s+\d+)?)\s*(?:M\d+芯片)?', r'\2 \1款', model_name)
    
    # 处理"22款iPad Pro" => "iPad Pro 2022款"
    model_name = re.sub(r'(\d{2})款(iPad\s+(?:Pro|Air|mini)(?:\s+\d+)?)', r'\2 20\1款', model_name)
    
    # 处理"iPad Pro M4 2024款" => "iPad Pro 2024款"
    model_name = re.sub(r'(iPad\s+(?:Pro|Air|mini)(?:\s+\d+)?)\s+M\d+\s+(\d{4})款', r'\1 \2款', model_name)
    
    # 处理可能残留的"M4芯片"等后缀
    model_name = re.sub(r'(iPad\s+(?:Pro|Air|mini)(?:\s+\d+)?(?:\s+\d{4}款))\s+M\d+芯片', r'\1', model_name)
    
    # 3. AirPods系列处理
    # 创建数字到中文数字的映射
    num_to_cn = {
        '1': '一',
        '2': '二',
        '3': '三',
        '4': '四',
        '5': '五',
        '6': '六',
        '7': '七',
        '8': '八',
        '9': '九',
        '10': '十'
    }
    
    # 3.1 处理AirPods Pro系列
    # 先处理数字代数到中文代数的转换: "AirPods Pro 2代" => "AirPods Pro 二代"
    airpods_match = re.search(r'AirPods\s+Pro\s+(\d+)(?:代|代|th)', model_name)
    if airpods_match:
        num = airpods_match.group(1)
        if num in num_to_cn:
            # 替换为"第X代"格式
            model_name = re.sub(r'(AirPods\s+Pro\s+)(\d+)(?:代|代|th)', 
                              fr'\1第{num_to_cn[num]}代', model_name)
    
    # 再处理已有的中文代数，确保有"第"字: "AirPods Pro 二代" => "AirPods Pro 第二代"
    model_name = re.sub(r'(AirPods\s+Pro\s+)(?!第)(一|二|三|四|五|六|七|八|九|十)代', 
                      r'\1第\2代', model_name)
    
    # 3.2 处理AirPods Max系列
    # 匹配普通的"AirPods Max"，将其视为第一代
    if re.search(r'AirPods\s+Max$', model_name) or re.search(r'AirPods\s+Max\s+(?!第)', model_name):
        model_name = re.sub(r'(AirPods\s+Max)(?:\s+|$)', r'\1 第一代', model_name)
    
    # 确保AirPods Max也采用"第X代"格式
    model_name = re.sub(r'(AirPods\s+Max\s+)(?!第)(一|二|三|四|五|六|七|八|九|十)代', 
                      r'\1第\2代', model_name)
    
    # 4. 处理Apple Watch系列格式问题
    # 4.1 修复"Apple Watch Series?X"的问题，确保是"Apple Watch Series X"
    # 移除所有可能出现的问号
    model_name = re.sub(r'(Apple\s+Watch\s+Series)\s*\?+\s*(\d+)', r'\1 \2', model_name)
    
    # 确保Apple Watch Series和数字之间有空格
    model_name = re.sub(r'(Apple\s+Watch\s+Series)(\d+)', r'\1 \2', model_name)
    
    # 4.2 修复Apple Watch Ultra格式问题
    # 处理"Apple Watch Ultra 22024"这种错误格式，改为"Apple Watch Ultra 2"
    model_name = re.sub(r'Apple\s+Watch\s+Ultra\s+(\d)(?:\d{4})', r'Apple Watch Ultra \1', model_name)
    
    # 处理其他常规Ultra格式
    model_name = re.sub(r'Apple\s+Watch\s+Ultra(\d+)', r'Apple Watch Ultra \1', model_name)
    model_name = re.sub(r'Apple\s+Watch\s*Ultra\s*(\d)\s*代?', r'Apple Watch Ultra \1', model_name)
    
    # 5. 修复"Apple Watch SE 2024款款"多了一个"款"字的问题
    model_name = re.sub(r'(Apple\s+Watch\s+SE\s+\d{4})款款', r'\1款', model_name)
    
    # 6. 通用修复所有可能重复的"款款"
    model_name = model_name.replace('款款', '款')
    
    # 7. 确保所有未匹配的Apple Watch都被正确格式化
    if "未匹配" in model_name and ("Apple Watch" in model_name):
        # 提取Apple Watch型号
        watch_match = re.search(r'(Apple\s+Watch\s+(?:Series\s+\d+|Ultra\s+\d+?|SE(?:\s+\d{4}款)?))', model_name)
        if watch_match:
            apple_watch_model = watch_match.group(1)
            # 检查是否包含年份，如果没有则不添加"款"
            if "SE" in apple_watch_model and not re.search(r'\d{4}款', apple_watch_model):
                apple_watch_model = re.sub(r'(Apple\s+Watch\s+SE)\s+(\d{4})', r'\1 \2款', apple_watch_model)
            return apple_watch_model
    
    return model_name

# 确保机型库中包含Apple Watch系列和AirPods系列
def update_device_library(device_library):
    # 检查是否有Apple Watch Series型号
    has_series_9 = any("Apple Watch Series 9" in model for model in device_library)
    has_series_10 = any("Apple Watch Series 10" in model for model in device_library)
    has_ultra_2 = any("Apple Watch Ultra 2" in model for model in device_library)
    
    # 检查是否有AirPods Max系列
    has_airpods_max_1 = any("AirPods Max 第一代" in model for model in device_library)
    has_airpods_max_2 = any("AirPods Max 第二代" in model for model in device_library)
    has_airpods_max_3 = any("AirPods Max 第三代" in model for model in device_library)
    
    # 检查是否有AirPods Pro系列
    has_airpods_pro_1 = any("AirPods Pro 第一代" in model for model in device_library)
    has_airpods_pro_2 = any("AirPods Pro 第二代" in model for model in device_library)
    has_airpods_pro_3 = any("AirPods Pro 第三代" in model for model in device_library)
    
    # 如果没有，添加到库中
    if not has_series_9:
        device_library.append("Apple Watch Series 9")
    if not has_series_10:
        device_library.append("Apple Watch Series 10")
    if not has_ultra_2:
        device_library.append("Apple Watch Ultra 2")
    
    # 添加AirPods Max系列
    if not has_airpods_max_1:
        device_library.append("AirPods Max 第一代")
    if not has_airpods_max_2:
        device_library.append("AirPods Max 第二代")
    if not has_airpods_max_3:
        device_library.append("AirPods Max 第三代")
    
    # 添加AirPods Pro系列
    if not has_airpods_pro_1:
        device_library.append("AirPods Pro 第一代")
    if not has_airpods_pro_2:
        device_library.append("AirPods Pro 第二代")
    if not has_airpods_pro_3:
        device_library.append("AirPods Pro 第三代")
    
    return device_library

# 更新机型库
device_library = update_device_library(device_library)
# 重新生成规范化的设备库
device_library_normalized = {model.lower().replace(" ", ""): model for model in device_library}
# 输出日志，查看标准化后的机型库内容
print("规范化的机型库样例:")
sample_entries = list(device_library_normalized.items())[:10]  # 输出前10个条目作为样例
for norm_key, original_model in sample_entries:
    print(f"{norm_key} -> {original_model}")

# 应用小新笔记本后处理函数
df['standard_model'] = df['standard_model'].apply(postprocess_xiaoxin_model)

# 应用通用格式后处理函数
df['standard_model'] = df['standard_model'].apply(format_model_name)

# 5. 统计数量和颜色分布
# 统计每个机型的总数
total_quantity = df.groupby('standard_model')['数量'].sum().reset_index()

# 统计每个机型的颜色分布
color_distribution = df.groupby(['standard_model', 'pro_color'])['数量'].sum().reset_index()

# 6. 处理未匹配的机型
unmatched = df[df['standard_model'].str.startswith('未匹配')]['cleaned_title'].unique()
if len(unmatched) > 0:
    print("以下机型未在机型库中匹配，请考虑添加到机型库：")
    for item in unmatched:
        print(item)

# 7. 保存结果
# 保存处理后的完整数据
df.to_csv('processed_phones.csv', index=False)
# 保存总数量统计
total_quantity.to_csv('total_quantity.csv', index=False)
# 保存颜色分布统计
color_distribution.to_csv('color_distribution.csv', index=False)

# 打印统计结果
print("\n每个机型的总数：")
print(total_quantity)
print("\n每个机型的颜色分布：")
print(color_distribution)