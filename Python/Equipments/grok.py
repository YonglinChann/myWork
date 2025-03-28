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
    "iPad Pro 2024款", "iPad Pro 2022款", 
    "iPad Air 2024款", "iPad 2022款", "iPad 2021款", 
    "iPad mini 6 2021款",
    
    # Apple Watch 系列
    "Apple Watch Ultra 2", "Apple Watch Series 9", "Apple Watch SE 2023款",
    
    # Apple 配件
    # AirPods 系列 - 通用匹配任意代数
    "AirPods 1", "AirPods 2", "AirPods 3", "AirPods 4", "AirPods 5", "AirPods 6",
    "AirPods Pro 第一代", "AirPods Pro 第二代",
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
    "Xiaomi 14 Ultra", "Xiaomi 14 Pro", "Xiaomi 14", "小米 14 Ultra", "小米14", "小米 13",
    "小米 Civi 4 Pro", "小米Civi 4 Pro",
    "小米 MIX Fold4", "小米 MIX Flip",
    "小米 Pad 6S Pro 12.4",
    
    # 红米系列
    "Redmi K70 Pro", "Redmi K70", "Redmi K70E", "红米K70至尊版", "红米 Turbo3",
    
    # vivo系列
    "vivo X100 Ultra", "vivo X100s Pro", "vivo X100s", "vivo X100 Pro", "vivo X100",
    "vivo X Fold3 Pro", "vivo S19 Pro", "vivo S19", "vivo S18e",
    "vivo iQOO 12 Pro", "vivo iQOO Neo9S Pro+", "vivo iQOO Neo9 Pro", "vivo iQOO Neo9",
    "vivo iQOO Z9 Turbo", "vivo Pad3 Pro",
    
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
    
    title = title.replace('5G全网通', '').replace('首月1元', '').replace('芝麻推广租物', '').replace('租物高分专享', '').replace('支持主动降噪', '').replace('通过率高', '').replace('顺丰发货', '').replace('顺丰包邮', '')
    title = title.replace('仅激活', '').replace('全新正品', '').replace('国行正品', '').replace('非监管机', '').strip()
    
    # 确保品牌名称前后有空格
    title = title.replace('华为', ' 华为 ').replace('小米', ' 小米 ').replace('三星', ' 三星 ').replace('  ', ' ').strip()
    return title


# 应用清洗函数
df['cleaned_title'] = df['pro_title'].apply(clean_title)


# 4. 匹配机型库
def match_device(title):
    """将清洗后的标题与机型库匹配"""
    # 规范化标题（小写并去除空格）
    normalized_title = title.lower().replace(" ", "")
    
    # 统一品牌名称处理
    normalized_title = normalized_title.replace("huawei", "华为").replace("huaweihuawe", "华为")
    normalized_title = normalized_title.replace("xiaomi", "小米").replace("xiaomixiaomi", "小米")
    normalized_title = normalized_title.replace("samsung", "三星").replace("三星三星", "三星")
    
    # 处理Apple品牌设备
    normalized_title = re.sub(r'(apple|苹果)\s*(iphone|ipad|watch|airpods)', r'\2', normalized_title, flags=re.IGNORECASE)
    
    # 处理OPPO Reno系列命名标准化
    reno_pattern = r'oppo\s*reno\s*(\d+)\s*(pro)?'
    reno_match = re.search(reno_pattern, normalized_title, re.IGNORECASE)
    if reno_match:
        number = reno_match.group(1)
        pro = reno_match.group(2)
        if pro:
            return f"OPPO Reno {number} Pro"
        return f"OPPO Reno {number}"
    
    # 处理华为Pura系列命名标准化
    pura_pattern = r'(?:华为\s*)?pura\s*(\d+)\s*((?:pro|ultra|pro\+))?'
    pura_match = re.search(pura_pattern, normalized_title, re.IGNORECASE)
    if pura_match:
        number = pura_match.group(1)
        suffix = pura_match.group(2)
        if suffix:
            suffix = suffix.upper().replace('PRO+', 'Pro+')
            return f"华为 Pura {number} {suffix}"
        return f"华为 Pura {number}"
    
    # 统一年份后缀格式
    normalized_title = re.sub(r'(\d{4})(?!款)', r'\1款', normalized_title)
    
    # 精确匹配
    if normalized_title in device_library_normalized:
        return device_library_normalized[normalized_title]

    # 部分匹配：检查机型库中的每个标准名称是否包含在标题中
    for norm_model, standard_model in device_library_normalized.items():
        # 对Apple iPad产品进行特殊处理
        if "ipad" in norm_model and "ipad" in normalized_title:
            # 提取iPad相关信息进行匹配
            # 移除芯片型号等额外信息
            normalized_title = re.sub(r'\s*m\d+芯片', '', normalized_title)
            
            # 标准化年份位置，将"2024款 iPad Pro"转换为"iPad Pro 2024款"
            normalized_title = re.sub(r'(\d{4})款\s*ipad\s*(.+)', r'ipad\2\1款', normalized_title)
            
            # 提取iPad后面的型号部分进行匹配
            ipad_model_pattern = r'ipad\s*(.+?)(?:\s+\d{4}款|$)'
            title_match = re.search(ipad_model_pattern, normalized_title)
            model_match = re.search(ipad_model_pattern, norm_model)
            
            if title_match and model_match:
                title_model = title_match.group(1)
                norm_model_part = model_match.group(1)
                
                # 移除多余的空格和额外信息
                title_model = re.sub(r'\s+', '', title_model)
                norm_model_part = re.sub(r'\s+', '', norm_model_part)
                
                # 提取年份
                year_match = re.search(r'(\d{4})款', normalized_title)
                if year_match:
                    year = year_match.group(1)
                    # 构建标准化的匹配字符串
                    standard_pattern = f"ipad{title_model}{year}款".lower()
                    norm_pattern = f"ipad{norm_model_part}{year}款".lower()
                    
                    if standard_pattern == norm_pattern:
                        return standard_model
                
                # 如果型号完全匹配，也返回结果
                if title_model == norm_model_part:
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

        # 对联想小新笔记本进行特殊处理
        elif "小新" in normalized_title:
            # 处理多年份组合的情况
            years_pattern = r'(\d{2})款/(\d{2})款小新(.+)'
            years_match = re.search(years_pattern, normalized_title)
            if years_match:
                year1, year2, model = years_match.groups()
                # 转换为完整年份
                full_year1 = f"20{year1}"
                full_year2 = f"20{year2}"
                # 处理Pro型号的空格
                model = re.sub(r'pro(\d+)', r'Pro \1', model, flags=re.IGNORECASE)
                # 构建标准化的设备名称
                return f"小新笔记本 {model.upper()} {full_year1}款/{full_year2}款"
        
        # 常规匹配
        elif norm_model in normalized_title:
            return standard_model

    # 如果没有匹配，返回原始标题并标记为未匹配
    return f"未匹配: {title}"


# 应用匹配函数
df['standard_model'] = df['cleaned_title'].apply(match_device)

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

# 更新设备库中的 Apple Watch 条目
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
    "iPad Pro 2024款", "iPad Pro 2022款", 
    "iPad Air 2024款", "iPad 2022款", "iPad 2021款", 
    "iPad mini 6 2021款",
    
    # Apple Watch 系列
    "Apple Watch Ultra 2", "Apple Watch Series 9", "Apple Watch SE 2023款",
    
    # Apple 配件
    # AirPods 系列 - 通用匹配任意代数
    "AirPods 1", "AirPods 2", "AirPods 3", "AirPods 4", "AirPods 5", "AirPods 6",
    "AirPods Pro 第一代", "AirPods Pro 第二代",
    # 华为系列
    "Huawei Mate 60 Pro", "华为 Mate 60 Pro+", "华为 Mate 60", "华为 Mate X5",
    "华为 Pura 70 Ultra", "华为 Pura 70 Pro+", "华为 Pura 70 Pro", "华为 Pura 70",
    "华为 nova 12 活力版", "华为 畅享70",
    "华为 MatePad Pro 11英寸 2022款",
    "华为 MateBook D16 SE 2024", "华为 MateBook D16 2024", 
    "华为 MateBook D14 2024", "华为 MateBook 14 2024款", "华为 MateBook E 2023",
    
    # 华为配件
    "HUAWEI WATCH 4 Pro", "HUAWEI WATCH GT 4", "HUAWEI WATCH GT 3 Pro",
    "华为 WATCH Ultimate", "华为 WATCH 4 Pro", "华为 WATCH GT 4", "华为Watch4",
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
    "vivo iQOO 12 Pro", "vivo iQOO Neo9S Pro+", "vivo iQOO Neo9 Pro", "vivo iQOO Neo9",
    "vivo iQOO Z9 Turbo", "vivo Pad3 Pro",
    
    # OPPO系列
    "OPPO Find X7 Ultra", "OPPO Find X7", "OPPO Find X6",
    "OPPO Reno 12 Pro", "OPPO Reno 12", "OPPO K12", "OPPO A3 Pro", "OPPO Watch 4 Pro",
    
    # 三星系列 
    "Samsung Galaxy S24 Ultra", "三星 Galaxy S24 Ultra",
    "三星 Galaxy Z Fold 6", "三星 Galaxy Z Fold 4", 
    "三星 Galaxy Z Flip 6", "三星 Galaxy Z Flip 4",
    
    # 荣耀系列
    "荣耀 Magic V3", "荣耀 Magic6 Pro", "荣耀 200 Pro", "荣耀 90 Pro",
    "荣耀 X50 GT", "荣耀 X50",
    
    # 努比亚系列
    "红魔9S Pro+", "红魔9S Pro", "红魔9Pro",
    
    # 一加系列
    "一加 Ace 3 Pro", "一加 Ace 3V", "一加 Ace 3", "一加平板 Pro",
    
    # realme系列
    "realme GT6",
    
    # 笔记本电脑
    "华硕 天选 5 Pro", "华硕 天选 5", "华硕 天选 4", "华硕 天选 Air 2024",
    "华硕 ROG 枪神 8", "华硕 无畏 Pro 15 2024", "华硕 灵耀 14 2024",
    "联想 拯救者 Y9000P 2024款", "联想 拯救者 Y9000X 2024款", "联想 拯救者 Y7000P 2024款",
    "联想 拯救者 R9000P 2024款", "联想 拯救者 R7000 2023款", "联想 拯救者 Y9000P 2023款",
    "联想 小新 Pro 16 2024", "联想 小新 Pro 14 2024", "联想 小新 16 2024", "联想 小新 16 2023 锐龙版",
    "联想 小新 Pad Pro 2025舒视版", "联想 小新 Pad 2024款",
    "ThinkBook 16+ 2024", "ThinkBook 16 2023", "ThinkBook 15 2023",
    "惠普 星Book Pro14 2024",
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
    title = re.sub(r'Apple\s*Watch\s*SE\s*(\d{4})', r'Apple Watch SE \1款', title)
    
    # 统一各品牌型号中的数字格式
    # OPPO Reno系列
    title = re.sub(r'OPPO\s*Reno(\d+)', r'OPPO Reno \1', title)
    
    # realme系列
    title = re.sub(r'realme\s*GT(\d+)', r'realme GT \1', title)
    
    # vivo系列
    title = re.sub(r'vivo\s*Pad(\d+)', r'vivo Pad \1', title)
    title = re.sub(r'vivo\s*X\s*Fold(\d+)', r'vivo X Fold \1', title)
    title = re.sub(r'vivo\s*iQOO\s*Neo(\d+)(\s*Pro)?', r'vivo iQOO Neo \1\2', title)
    
    # 一加系列
    title = re.sub(r'一加\s*Ace(\d+)', r'一加 Ace \1', title)
    
    # 三星系列
    title = re.sub(r'(三星|Samsung)\s*Galaxy\s*Z\s*Flip(\d+)', r'\1 Galaxy Z Flip \2', title)
    title = re.sub(r'(三星|Samsung)\s*Galaxy\s*Z\s*Fold(\d+)', r'\1 Galaxy Z Fold \2', title)
    
    title = title.replace('5G全网通', '').replace('首月1元', '').replace('芝麻推广租物', '').replace('租物高分专享', '').replace('支持主动降噪', '').replace('通过率高', '').replace('顺丰发货', '').replace('顺丰包邮', '').replace('闪充拍照全新手机', '')
    title = title.replace('仅激活', '').replace('全新正品', '').replace('国行正品', '').replace('非监管机', '').strip()
    # 去除新旧程度描述
    title = re.sub(r'\d+新', '', title) 
    title = title.replace('全新', '').replace('准新', '').replace('二手', '').replace('翻新', '')
    # 确保品牌名称前后有空格
    title = title.replace('华为', ' 华为 ').replace('小米', ' 小米 ').replace('三星', ' 三星 ').replace('  ', ' ').strip()
    return title


# 应用清洗函数
df['cleaned_title'] = df['pro_title'].apply(clean_title)


# 4. 匹配机型库
def match_device(title):
    """将清洗后的标题与机型库匹配"""
    # 规范化标题（小写并去除空格）
    normalized_title = title.lower().replace(" ", "")
    
    # 统一品牌名称处理
    normalized_title = normalized_title.replace("huawei", "华为").replace("huaweihuawe", "华为")
    normalized_title = normalized_title.replace("xiaomi", "小米").replace("xiaomixiaomi", "小米")
    normalized_title = normalized_title.replace("samsung", "三星").replace("三星三星", "三星")
    
    # 处理Apple品牌设备
    normalized_title = re.sub(r'(apple|苹果)\s*(iphone|ipad|watch|airpods)', r'\2', normalized_title, flags=re.IGNORECASE)
    
    # 处理OPPO Reno系列命名标准化
    reno_pattern = r'oppo\s*reno\s*(\d+)\s*(pro)?'
    reno_match = re.search(reno_pattern, normalized_title, re.IGNORECASE)
    if reno_match:
        number = reno_match.group(1)
        pro = reno_match.group(2)
        if pro:
            return f"OPPO Reno {number} Pro"
        return f"OPPO Reno {number}"
    
    # 处理华为Pura系列命名标准化
    pura_pattern = r'(?:华为\s*)?pura\s*(\d+)\s*((?:pro|ultra|pro\+))?'
    pura_match = re.search(pura_pattern, normalized_title, re.IGNORECASE)
    if pura_match:
        number = pura_match.group(1)
        suffix = pura_match.group(2)
        if suffix:
            suffix = suffix.upper().replace('PRO+', 'Pro+')
            return f"华为 Pura {number} {suffix}"
        return f"华为 Pura {number}"
    
    # 统一年份后缀格式
    normalized_title = re.sub(r'(\d{4})(?!款)', r'\1款', normalized_title)
    
    # 精确匹配
    if normalized_title in device_library_normalized:
        return device_library_normalized[normalized_title]

    # 部分匹配：检查机型库中的每个标准名称是否包含在标题中
    for norm_model, standard_model in device_library_normalized.items():
        # 对Apple iPad产品进行特殊处理
        if "ipad" in norm_model and "ipad" in normalized_title:
            # 提取iPad相关信息进行匹配
            # 移除芯片型号等额外信息
            normalized_title = re.sub(r'\s*m\d+芯片', '', normalized_title)
            
            # 标准化年份位置，将"2024款 iPad Pro"转换为"iPad Pro 2024款"
            normalized_title = re.sub(r'(\d{4})款\s*ipad\s*(.+)', r'ipad\2\1款', normalized_title)
            
            # 提取iPad后面的型号部分进行匹配
            ipad_model_pattern = r'ipad\s*(.+?)(?:\s+\d{4}款|$)'
            title_match = re.search(ipad_model_pattern, normalized_title)
            model_match = re.search(ipad_model_pattern, norm_model)
            
            if title_match and model_match:
                title_model = title_match.group(1)
                norm_model_part = model_match.group(1)
                
                # 移除多余的空格和额外信息
                title_model = re.sub(r'\s+', '', title_model)
                norm_model_part = re.sub(r'\s+', '', norm_model_part)
                
                # 提取年份
                year_match = re.search(r'(\d{4})款', normalized_title)
                if year_match:
                    year = year_match.group(1)
                    # 构建标准化的匹配字符串
                    standard_pattern = f"ipad{title_model}{year}款".lower()
                    norm_pattern = f"ipad{norm_model_part}{year}款".lower()
                    
                    if standard_pattern == norm_pattern:
                        return standard_model
                
                # 如果型号完全匹配，也返回结果
                if title_model == norm_model_part:
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

        # 对联想小新笔记本进行特殊处理
        elif "小新" in normalized_title:
            # 处理多年份组合的情况
            years_pattern = r'(\d{2})款/(\d{2})款小新(.+)'
            years_match = re.search(years_pattern, normalized_title)
            if years_match:
                year1, year2, model = years_match.groups()
                # 转换为完整年份
                full_year1 = f"20{year1}"
                full_year2 = f"20{year2}"
                # 处理Pro型号的空格
                model = re.sub(r'pro(\d+)', r'Pro \1', model, flags=re.IGNORECASE)
                # 构建标准化的设备名称
                return f"小新笔记本 {model.upper()} {full_year1}款/{full_year2}款"
        
        # 常规匹配
        elif norm_model in normalized_title:
            return standard_model

    # 如果没有匹配，返回原始标题并标记为未匹配
    return f"未匹配: {title}"


# 应用匹配函数
df['standard_model'] = df['cleaned_title'].apply(match_device)

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