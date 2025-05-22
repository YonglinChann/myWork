from picColor_scale_first_all import process_image_combined

if __name__ == "__main__":
    # 示例：处理一张图片
    imageName = "Maggie-1"    
    targetDir = f"/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/Pic_Color_Select【成品】/1/pic"
    
    # 调用封装函数处理图片
    combined_results = process_image_combined(imageName, targetDir)
    
    if combined_results:
        print(f"整合后的结果包含 {len(combined_results)} 个图片的数据")
    else:
        print("图片处理失败，没有生成有效的结果数据。")