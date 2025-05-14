import picColor_scale_first as pic_default
import picColor_scale_first_T as pic_T
import picColor_scale_first_C as pic_C


if __name__ == "__main__":
    # 示例：处理一张图片
    input_image = "/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/Pic_Color_Select【成品】/1/pic/2024-23.jpg"  # 输入图片路径
    output_image = "/Users/chenyonglin/myCode/gitee/myWork/Python/Pic_Color/Pic_Color_Select【成品】/1/pic/2024-23_O.png"  # 输出图片路径
    output_data = "data.txt"  # 输出数据包文件路径
        # 调用函数处理图片
    result = process_image_for_eink(input_image, output_image)
    #result = process_image_for_eink(input_image, output_image, output_data)
    if result:
        print(f"图片处理成功，生成了 {len(result)} 个数据包")
    else:
        print("图片处理失败，有可能路径中没有该图片。")




