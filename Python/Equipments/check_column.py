import pandas as pd

# 加载数据文件
data_df = pd.read_excel("新合并.xlsx")

# 检查列名
print("文件列名:", data_df.columns.tolist())

# 如果存在"机型数据"列，查看前5行内容
if "机型数据" in data_df.columns:
    print("\n机型数据列前5行内容:")
    for i, value in enumerate(data_df["机型数据"].head(5)):
        print(f"{i+1}. {value}")
    
    # 显示与前三列的对比
    print("\n对比前三列与机型数据列:")
    sample = data_df.head(5)
    for i, row in sample.iterrows():
        print(f"行 {i+1}:")
        if "pro_title" in data_df.columns:
            print(f"  pro_title: {row['pro_title']}")
        if "品牌" in data_df.columns:
            print(f"  品牌: {row['品牌']}")
        if "pro_model" in data_df.columns:
            print(f"  pro_model: {row['pro_model']}")
        print(f"  机型数据: {row['机型数据']}")
        print("-" * 50)
else:
    print("文件中不存在'机型数据'列") 