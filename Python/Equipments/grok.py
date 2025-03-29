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
    "华为 nova 12 活力版", "华为 畅享 70",
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
    "vivo X90 Pro+", "vivo X90 Pro", "vivo X90",
    "vivo X80 Pro", "vivo X80", 
    "vivo X Fold 3 Pro", "vivo X Fold 3", "vivo X Fold 2", "vivo X Fold",
    "vivo X Flip", "vivo X Flip 2", "vivo X Flip 3",
    "vivo S19 Pro", "vivo S19", "vivo S18 Pro", "vivo S18", "vivo S18e", "vivo S20", "vivo S20 Pro",
    "vivo iQOO 12 Pro", "vivo iQOO 12", "vivo iQOO 11 Pro", "vivo iQOO 11", "vivo iQOO 10 Pro", "vivo iQOO 10",
    "vivo iQOO Neo 10 Pro", "vivo iQOO Neo 10", "vivo iQOO Neo 9 Pro", "vivo iQOO Neo 9", "vivo iQOO Neo9S Pro+",
    "vivo iQOO Z9 Turbo", "vivo iQOO Z9", "vivo iQOO Z9x",
    "vivo Y300 Pro", "vivo Y300", "vivo Y200i", "vivo Y200",
    "vivo Pad 3 Pro", "vivo Pad 3", "vivo Pad 2",
    
    # OPPO系列
    "OPPO Find X7 Ultra", "OPPO Find X7", "OPPO Find X6",
    "OPPO Reno 12 Pro", "OPPO Reno 12", "OPPO K12", "OPPO A3 Pro", "OPPO Watch 4 Pro",
    
    # 三星系列
    "Samsung Galaxy S24 Ultra", "三星 Galaxy S24 Ultra",
    "三星 Galaxy Z Fold 5", "三星 Galaxy Z Fold 4", 
    "三星 Galaxy Z Flip 5", "三星 Galaxy Z Flip 4",
    
    # 荣耀系列
    "荣耀Magic V3", "荣耀 Magic 6 Pro", "荣耀 200 Pro",
    "荣耀 X50 GT", "荣耀 X50",
    
    # 努比亚系列
    "红魔9S Pro+", "红魔9S Pro", "红魔9Pro",
    
    # 一加系列
    "一加 Ace 3 Pro", "一加 Ace 3V", "一加 Ace 3", "一加平板 Pro",
    
    # realme系列
    "realme GT 6",
    
    # 笔记本电脑
    "华硕 天选 5 Pro", "华硕 天选 5", "华硕 天选 4", "华硕 天选 Air 2024",
    "华硕 ROG 枪神 8", "华硕 无畏 Pro 15 2024", "华硕 灵耀 14 2024",
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
    
    # 统一品牌名称
    title = re.sub(r'(华为|HUAWEI|Huawei)\s*', '华为 ', title)
    title = re.sub(r'(小米|Xiaomi|MI)\s*', '小米 ', title)
    title = re.sub(r'(三星|Samsung)\s*', '三星 ', title)
    title = re.sub(r'(vivo|VIVO)\s*', 'vivo ', title)
    title = re.sub(r'(IQOO|Iqoo|iQoo)\s*', 'iQOO ', title)
    title = re.sub(r'(OPPO|oppo)\s*', 'OPPO ', title)
    title = re.sub(r'(惠普|HP|hp)\s*', '惠普 ', title)
    title = re.sub(r'(华硕|ASUS|asus)\s*', '华硕 ', title)
    
    # 处理重复的品牌名称
    title = re.sub(r'小米\s+小米\s+', '小米 ', title)
    title = re.sub(r'华为\s+华为\s+', '华为 ', title)
    title = re.sub(r'惠普\s+惠普\s+', '惠普 ', title)
    title = re.sub(r'华硕\s+华硕\s+', '华硕 ', title)
    # 处理OPPO Find系列
    title = re.sub(r'OPPO\s*FIND\s*([NX]\d+)', r'OPPO Find \1', title, flags=re.IGNORECASE)
    title = re.sub(r'(荣耀|HONOR|Honor)\s*', '荣耀 ', title)
    title = re.sub(r'(realme|Realme|REALME)\s*', 'realme 真我 ', title)
    title = re.sub(r'(努比亚|NUBIA|Nubia)\s*', '努比亚 ', title)
    title = re.sub(r'(一加|OnePlus|Oneplus)\s*', '一加 ', title)
    title = re.sub(r'(红魔|RedMagic|Redmagic)\s*', '红魔 ', title)

    # 统一华为品牌名称
    title = re.sub(r'(华为|HUAWEI|Huawei)\s*', '华为 ', title)
    
    # 统一小米品牌名称
    title = re.sub(r'(小米|Xiaomi|MI)\s*', '小米 ', title)
    
    # 统一三星品牌名称
    title = re.sub(r'(三星|Samsung)\s*', '三星 ', title)
    
    # 统一vivo和iQOO品牌名称
    title = re.sub(r'(vivo|VIVO)\s*', 'vivo ', title)
    title = re.sub(r'(IQOO|Iqoo|iQoo)\s*', 'iQOO ', title)
    
    # 统一年份后缀格式
    title = re.sub(r'(\d{4})(?!\s*款)', r'\1款', title)
    
    # 统一 Apple Watch 系列命名
    # 处理 Series 系列
    title = re.sub(r'Apple\s*Watch\s*S(?:eries)?\s*(\d+)', r'Apple Watch Series \1', title)
    # 处理 Ultra 系列
    title = re.sub(r'Apple\s*Watch\s*Ultra\s*(\d+)', r'Apple Watch Ultra \1', title)
    # 处理 SE 系列（保持年份格式）
    title = re.sub(r'Apple\s*Watch\s*SE\s*(\d{4})', r'Apple Watch SE \1款', title)
    
    # 统一处理三星Galaxy Z系列
    # 处理Galaxy Z Flip系列
    title = re.sub(r'(Galaxy\s+Z\s+Flip)(\d+)', r'\1 \2', title)
    # 处理Galaxy Z Fold系列
    title = re.sub(r'(Galaxy\s+Z\s+Fold)(\d+)', r'\1 \2', title)
    
    # 统一处理vivo和iQOO的命名格式
    # 处理vivo X系列
    title = re.sub(r'(vivo\s+X)(\d+)(?!\s+)', r'\1\2 ', title)
    # 处理vivo S系列
    title = re.sub(r'(vivo\s+S)(\d+)(?!\s+)', r'\1\2 ', title)
    # 处理iQOO系列
    title = re.sub(r'(iQOO\s+)(\d+)(?!\s+)', r'\1\2 ', title)
    title = re.sub(r'(iQOO\s+Neo\s+)(\d+)(?!\s+)', r'\1\2 ', title)
    
    title = title.replace('5G全网通', '').replace('首月1元', '').replace('芝麻推广租物', '').replace('租物高分专享', '').replace('支持主动降噪', '').replace('通过率高', '').replace('顺丰发货', '').replace('顺丰包邮', '').replace('橡胶表带', '').replace('智能手表', '')
    title = title.replace('仅激活', '').replace('全新正品', '').replace('国行正品', '').replace('非监管机', '').strip()
    
    # 确保品牌名称前后有空格
    title = title.replace('华为', ' 华为 ').replace('小米', ' 小米 ').replace('三星', ' 三星 ').replace('vivo', ' vivo ').replace('  ', ' ').strip()
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
    normalized_title = normalized_title.replace("xiaomi", "小米").replace("xiaomixiaomi", "小米").replace("小米小米", "小米")
    normalized_title = normalized_title.replace("samsung", "三星").replace("三星三星", "三星")
    normalized_title = normalized_title.replace("oppo", "OPPO").replace("OPPOOPPO", "OPPO")
    normalized_title = normalized_title.replace("hp", "惠普").replace("惠普惠普", "惠普")
    normalized_title = normalized_title.replace("asus", "华硕").replace("华硕华硕", "华硕")
    
    # 处理荣耀设备
    honor_pattern = r'^荣耀\s*([a-z0-9]+(?:\s*[a-z0-9]+)*(?:\s*pro|\s*plus|\s*\+)?)'  
    honor_match = re.search(honor_pattern, title, re.IGNORECASE)
    if honor_match:
        model = honor_match.group(1).strip()
        return f"荣耀 {model}".strip()
    
    # 处理雷神设备
    thunder_pattern = r'^雷神\s*([a-z0-9]+(?:\s*[a-z0-9]+)*(?:\s*pro|\s*plus|\s*\+)?)'  
    thunder_match = re.search(thunder_pattern, title, re.IGNORECASE)
    if thunder_match:
        model = thunder_match.group(1).strip()
        return f"雷神 {model}".strip()
    
    # 处理一加设备
    oneplus_pattern = r'一加\s*(\d+|ace\s*\d+|pad)\s*((?:pro|ultra|\+|plus|v)?)'  
    oneplus_match = re.search(oneplus_pattern, title, re.IGNORECASE)
    if oneplus_match:
        model = oneplus_match.group(1).strip()
        suffix = oneplus_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
            elif suffix.lower() == "v": suffix = "V"
        # 处理Ace系列
        if model.lower().startswith("ace"):
            model = f"Ace {model[3:]}"
        # 处理Pad系列
        elif model.lower() == "pad":
            model = "平板"
        return f"一加 {model} {suffix}".strip()

    # 处理realme设备
    realme_pattern = r'realme\s*([a-z0-9]+(?:\s*[a-z0-9]+)*(?:\s*pro|\s*plus|\s*\+)?)'  
    realme_match = re.search(realme_pattern, title, re.IGNORECASE)
    if realme_match:
        model = realme_match.group(1).strip()
        return f"realme 真我 {model}".strip()

    # 处理努比亚设备
    nubia_pattern = r'努比亚\s*([a-z0-9]+(?:\s*[a-z0-9]+)*(?:\s*pro|\s*plus|\s*\+)?)'  
    nubia_match = re.search(nubia_pattern, title, re.IGNORECASE)
    if nubia_match:
        model = nubia_match.group(1).strip()
        return f"努比亚 {model}".strip()

    # 处理魅族设备
    meizu_pattern = r'^魅族\s*([a-z0-9]+(?:\s*[a-z0-9]+)*(?:\s*pro|\s*plus|\s*\+)?)'  
    meizu_match = re.search(meizu_pattern, title, re.IGNORECASE)
    if meizu_match:
        model = meizu_match.group(1).strip()
        return f"魅族 {model}".strip()
    
    # 处理Redmi设备
    redmi_pattern = r'^redmi\s*note\s*(\d+)\s*'  
    redmi_match = re.search(redmi_pattern, title, re.IGNORECASE)
    if redmi_match:
        number = redmi_match.group(1)
        return f"Redmi Note {number}".strip()
    
    # 处理联想ThinkBook和ThinkPad设备
    think_pattern = r'^(thinkbook|thinkpad)\s+([^\s]+(?:\s+[^\s]+)*)'  
    think_match = re.search(think_pattern, title, re.IGNORECASE)
    if think_match:
        series = think_match.group(1)
        model = think_match.group(2)
        return f"联想 {series} {model}".strip()
    
    # 处理Apple品牌设备
    if "apple" in normalized_title.lower() or "苹果" in normalized_title:
        # 移除Apple/苹果前缀，保留设备名称
        normalized_title = re.sub(r'(apple|苹果)\s*', '', normalized_title, flags=re.IGNORECASE)
    
    # 处理三星设备
    # 处理S系列
    s_pattern = r'三星\s*三星?\s*s(\d+)\s*((?:ultra|\+|plus|pro|fe))?'
    s_match = re.search(s_pattern, normalized_title, re.IGNORECASE)
    if s_match:
        number = s_match.group(1)
        suffix = s_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
            elif suffix.lower() == "fe": suffix = "FE"
        return f"三星 S{number} {suffix}".strip()

    # 处理W系列
    w_pattern = r'三星\s*w(\d+)\s*((?:flip))?'
    w_match = re.search(w_pattern, normalized_title, re.IGNORECASE)
    if w_match:
        number = w_match.group(1)
        suffix = w_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
        return f"三星 W{number} {suffix}".strip()

    # 处理Galaxy系列设备
    # 处理Z系列
    z_pattern = r'galaxy\s*z\s*(flip|fold)\s*(\d+)?\s*((?:ultra|\+|plus|pro|fe))?'
    z_match = re.search(z_pattern, normalized_title, re.IGNORECASE)
    if z_match:
        model_type = z_match.group(1).capitalize()
        number = z_match.group(2) or ""
        suffix = z_match.group(3) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
            elif suffix.lower() == "fe": suffix = "FE"
        return f"三星 Galaxy Z {model_type} {number} {suffix}".strip()

    # 处理Galaxy Tab系列
    tab_pattern = r'galaxy\s*tab\s*([a-z]\d+(?:\s*\+)?|\d+)\s*((?:ultra|\+|plus|pro|fe))?'
    tab_match = re.search(tab_pattern, normalized_title, re.IGNORECASE)
    if tab_match:
        model = tab_match.group(1).strip()
        suffix = tab_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
            elif suffix.lower() == "fe": suffix = "FE"
        return f"三星 Galaxy Tab {model.upper()} {suffix}".strip()

    # 处理其他Galaxy系列设备
    galaxy_pattern = r'galaxy\s*([a-z]\d+)\s*((?:ultra|\+|plus|pro|fe))?'
    galaxy_match = re.search(galaxy_pattern, normalized_title, re.IGNORECASE)
    if galaxy_match:
        model = galaxy_match.group(1).strip()
        suffix = galaxy_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
            elif suffix.lower() == "fe": suffix = "FE"
        return f"三星 Galaxy {model.upper()} {suffix}".strip()
    
    # 处理MatePad系列设备
    matepad_pattern = r'matepad\s*(pro)?\s*(\d{4})?(?:年|款)?'
    matepad_match = re.search(matepad_pattern, normalized_title, re.IGNORECASE)
    if matepad_match:
        model_type = matepad_match.group(1) or ""
        year = matepad_match.group(2) or ""
        if model_type:
            model_type = "Pro"
        if year:
            return f"华为 MatePad {model_type} {year}款".strip()
        return f"华为 MatePad {model_type}".strip()
    
    # 处理OPPO设备
    # 优先处理Find系列
    find_pattern = r'oppo\s*find\s*([nx]\s*\d+)\s*((?:pro|ultra|\+|plus))?'
    find_match = re.search(find_pattern, normalized_title, re.IGNORECASE)
    if find_match:
        model = find_match.group(1).strip().upper()
        suffix = find_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
        return f"OPPO Find {model} {suffix}".strip()

    # 处理OPPO Pad和Watch系列
    oppo_special_pattern = r'oppo\s*(pad|watch)\s*(\d+)?\s*((?:pro|ultra|\+|plus|air))?'
    oppo_special_match = re.search(oppo_special_pattern, normalized_title, re.IGNORECASE)
    if oppo_special_match:
        device_type = oppo_special_match.group(1).capitalize()
        number = oppo_special_match.group(2) or ""
        suffix = oppo_special_match.group(3) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
            elif suffix.lower() == "air": suffix = "Air"
        return f"OPPO {device_type} {number} {suffix}".strip()

    # 处理其他OPPO设备
    oppo_pattern = r'oppo\s*([a-z]\d+)\s*((?:pro|ultra|\+|plus))?'
    oppo_match = re.search(oppo_pattern, normalized_title, re.IGNORECASE)
    if oppo_match:
        model = oppo_match.group(1).strip()
        suffix = oppo_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
        return f"OPPO {model.upper()} {suffix}".strip()
    
    # 处理vivo和iQOO系列
    if "vivo" in normalized_title.lower() or "iqoo" in normalized_title.lower():
        # 处理vivo X系列
        x_pattern = r'(?:vivo\s*)?x(?:fold|flip)?(\d+)(?:\s*(pro|ultra|\+|plus|pro\+))?'
        x_match = re.search(x_pattern, normalized_title, re.IGNORECASE)
        if x_match:
            number = x_match.group(1)
            suffix = x_match.group(2) or ""
            
            # 处理不同后缀
            if suffix.lower() in ["plus", "+"]:
                suffix = "Plus"
            elif suffix.lower() == "pro+":
                suffix = "Pro+"
            elif suffix:
                suffix = suffix.capitalize()
            
            # 处理特殊情况: X Fold和X Flip系列
            if "fold" in normalized_title.lower():
                gen_num = number if number else ""
                return f"vivo X Fold{gen_num} {suffix}".strip()
            elif "flip" in normalized_title.lower():
                gen_num = number if number else ""
                return f"vivo X Flip{gen_num} {suffix}".strip()
            else:
                return f"vivo X{number} {suffix}".strip()
        
        # 处理vivo S系列
        s_pattern = r'(?:vivo\s*)?s(\d+)(?:\s*(pro|e))?'
        s_match = re.search(s_pattern, normalized_title, re.IGNORECASE)
        if s_match:
            number = s_match.group(1)
            suffix = s_match.group(2) or ""
            
            # 统一大小写
            if suffix:
                suffix = suffix.capitalize() if suffix.lower() != 'e' else suffix.lower()
            
            return f"vivo S{number} {suffix}".strip()
        
        # 处理iQOO系列
        iqoo_pattern = r'(?:vivo\s*)?iqoo(?:\s*neo)?(?:\s*(\d+))?(?:\s*(pro|ultra|pro\+|turbo))?'
        iqoo_match = re.search(iqoo_pattern, normalized_title, re.IGNORECASE)
        if iqoo_match:
            number = iqoo_match.group(1) or ""
            suffix = iqoo_match.group(2) or ""
            
            # 统一大小写
            if suffix:
                if suffix.lower() == "pro+":
                    suffix = "Pro+"
                elif suffix.lower() == "turbo":
                    suffix = "Turbo"
                else:
                    suffix = suffix.capitalize()
            
            # 判断是Neo系列还是普通iQOO
            if "neo" in normalized_title.lower():
                return f"vivo iQOO Neo {number} {suffix}".strip()
            else:
                return f"vivo iQOO {number} {suffix}".strip()
    
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
    
    # 处理MacBook系列
    if "macbook" in normalized_title.lower():
        # 处理MacBook Air系列
        macbook_pattern = r'macbook\s*(air|pro)(?:\s*m\d+)?(?:\s*(\d{4})(?:年|款)?)?'
        macbook_match = re.search(macbook_pattern, normalized_title, re.IGNORECASE)
        if macbook_match:
            model_type = macbook_match.group(1).capitalize()
            year = macbook_match.group(2)
            
            # 构建标准格式
            if year:
                return f"MacBook {model_type} {year}款"
            else:
                return f"MacBook {model_type}"
    
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
    
    # 处理Apple品牌设备
    normalized_title = re.sub(r'(apple|苹果)\s*(iphone|ipad|watch|airpods)', r'\2', normalized_title, flags=re.IGNORECASE)
    
    # 处理OPPO设备命名标准化
    # 处理Find系列
    find_pattern = r'oppo\s*find\s*([nx]\s*\d+)\s*((?:pro|ultra|\+|plus))?'
    find_match = re.search(find_pattern, normalized_title, re.IGNORECASE)
    if find_match:
        model = find_match.group(1).strip().upper()
        suffix = find_match.group(2) or ""
        if suffix:
            suffix = suffix.strip().capitalize()
            if suffix.lower() in ["plus", "+"]: suffix = "Plus"
        return f"OPPO Find {model} {suffix}".strip()

    # 处理Reno系列
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

    # 2.1 处理MacBook系列格式问题
    # 修复"MacBook Air 2020款年款"为"MacBook Air 2020款"
    model_name = re.sub(r'(MacBook\s+(?:Air|Pro)\s+\d{4})款年款', r'\1款', model_name)
    
    # 处理"99新 MacBook Air 2022款 MacBook Air 13寸"这种重复的情况
    model_name = re.sub(r'(?:新|九成新|95新|99新)\s+(MacBook\s+(?:Air|Pro).+?MacBook.+)', r'\1', model_name)
    model_name = re.sub(r'(MacBook\s+(?:Air|Pro)\s+\d{4}款).+MacBook.+', r'\1', model_name)
    
    # 标准化MacBook年份格式
    model_name = re.sub(r'(MacBook\s+(?:Air|Pro))\s+(\d{4})\s*(?:款)?', r'\1 \2款', model_name)
    model_name = re.sub(r'(MacBook\s+(?:Air|Pro))\s+M\d+\s+(\d{4})\s*(?:款)?', r'\1 \2款', model_name)
    
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
    
    # 7. 处理三星Galaxy Z系列命名规范
    # 确保Galaxy Z Flip和数字之间有空格: "Galaxy Z Flip6" => "Galaxy Z Flip 6"
    model_name = re.sub(r'(Galaxy\s+Z\s+Flip)(\d+)', r'\1 \2', model_name, flags=re.IGNORECASE)
    # 确保Galaxy Z Fold和数字之间有空格: "Galaxy Z Fold6" => "Galaxy Z Fold 6"
    model_name = re.sub(r'(Galaxy\s+Z\s+Fold)(\d+)', r'\1 \2', model_name, flags=re.IGNORECASE)
    
    # 8. 处理vivo系列命名标准化
    # 8.1 处理vivo X系列
    model_name = re.sub(r'(vivo\s+X)(\d+)(?!\s+)', r'\1\2 ', model_name)
    # 处理vivo X100s的情况，确保无多余空格："vivo X100 s" => "vivo X100s"
    model_name = re.sub(r'vivo\s+X(\d+)\s+s(?:\s+|$)', r'vivo X\1s ', model_name)
    
    # 修复"vivo X10 0 Pro"和"vivo X10 0 Ultra"的空格问题
    model_name = re.sub(r'vivo\s+X(\d+)\s+(\d+)(?:\s+(Pro|Ultra))?', r'vivo X\1\2 \3', model_name)
    
    # 8.2 处理vivo S系列
    model_name = re.sub(r'(vivo\s+S)(\d+)(?!\s+)', r'\1\2 ', model_name)
    # 修复"vivo S1 8 e"或"vivo S1 9 Pro"格式问题
    model_name = re.sub(r'vivo\s+S(\d+)\s+(\d+)(?:\s+(e|Pro|Ultra))?', r'vivo S\1\2\3 ', model_name)
    
    # 8.3 处理vivo iQOO系列
    # 普通iQOO系列
    model_name = re.sub(r'(vivo\s+iQOO\s*)(\d+)(?!\s+)', r'\1\2 ', model_name)
    # 修复"vivo iQOO 1 1"格式问题
    model_name = re.sub(r'vivo\s+iQOO\s+(\d+)\s+(\d+)(?:\s+(Pro|Ultra|\+|Plus))?', r'vivo iQOO \1\2 \3', model_name)
    
    # Neo系列
    model_name = re.sub(r'(vivo\s+iQOO\s*Neo\s*)(\d+)(?!\s+)', r'\1\2 ', model_name)
    # 修复"vivo iQOO Neo 9 Pro"格式问题，去除数字之间的空格
    model_name = re.sub(r'vivo\s+iQOO\s+Neo\s+(\d+)\s+(\d+)?(?:\s+(Pro|Ultra|\+|Plus|S))?', r'vivo iQOO Neo \1\2\3 ', model_name)
    # 修复"vivo iQOO Neo9S Pro+"格式
    model_name = re.sub(r'vivo\s+iQOO\s+Neo(\d+)S\s+(Pro\+|Pro)', r'vivo iQOO Neo\1S \2', model_name)
    
    # Z系列
    model_name = re.sub(r'(vivo\s+iQOO\s*Z)(\d+)(?!\s+)', r'\1\2 ', model_name)
    # 修复"vivo iQOO Z9x"格式问题
    model_name = re.sub(r'vivo\s+iQOO\s+Z(\d+)x', r'vivo iQOO Z\1x', model_name)
    
    # 8.4 确保型号后缀格式正确 (Pro, Ultra, Pro+等)
    # 普通Pro后缀
    model_name = re.sub(r'(vivo\s+(?:X|S|iQOO|iQOO\s+Neo|iQOO\s+Z)\d+\s+)pro\b', r'\1Pro', model_name, flags=re.IGNORECASE)
    # Ultra后缀
    model_name = re.sub(r'(vivo\s+(?:X|S|iQOO|iQOO\s+Neo|iQOO\s+Z)\d+\s+)ultra\b', r'\1Ultra', model_name, flags=re.IGNORECASE)
    # Pro+后缀
    model_name = re.sub(r'(vivo\s+(?:X|S|iQOO|iQOO\s+Neo|iQOO\s+Z)\d+\s+)pro\+', r'\1Pro+', model_name, flags=re.IGNORECASE)
    
    # 8.5 清理多余的空格，确保每个部分之间只有一个空格
    model_name = re.sub(r'\s+', ' ', model_name).strip()
    
    # 8.6 为Pad系列标准化空格
    model_name = re.sub(r'(vivo\s+Pad)(\d+)\s+Pro', r'\1 \2 Pro', model_name)
    # 修复"vivo Pad 3 Pro"格式
    model_name = re.sub(r'vivo\s+Pad(\d+)\s+Pro', r'vivo Pad \1 Pro', model_name)
    
    # 9. 确保所有未匹配的Apple Watch都被正确格式化
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
    
    # 检查是否有MacBook系列
    has_macbook_air_2020 = any("MacBook Air 2020款" in model for model in device_library)
    has_macbook_air_2022 = any("MacBook Air 2022款" in model for model in device_library)
    has_macbook_air_2024 = any("MacBook Air 2024款" in model for model in device_library)
    has_macbook_pro_2024 = any("MacBook Pro 2024款" in model for model in device_library)
    
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
    
    # 添加MacBook系列
    if not has_macbook_air_2020:
        device_library.append("MacBook Air 2020款")
    if not has_macbook_air_2022:
        device_library.append("MacBook Air 2022款")
    if not has_macbook_air_2024:
        device_library.append("MacBook Air 2024款")
    if not has_macbook_pro_2024:
        device_library.append("MacBook Pro 2024款")
    
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