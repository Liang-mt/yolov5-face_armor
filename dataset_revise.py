import os
import shutil


def delete_unmatched_txt_files(img_folder, txt_folder, img_extensions=None):
    """
    删除文本文件夹中没有对应图片文件的.txt文件

    Args:
        img_folder (str): 图片文件夹路径
        txt_folder (str): 文本文件夹路径
        img_extensions (list): 要识别的图片扩展名，默认包含常见格式
    """
    # 默认支持的图片格式
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

    # 验证文件夹是否存在
    if not os.path.exists(img_folder):
        print(f"错误：图片文件夹 {img_folder} 不存在！")
        return
    if not os.path.exists(txt_folder):
        print(f"错误：文本文件夹 {txt_folder} 不存在！")
        return

    # 1. 提取所有图片文件名（不含扩展名）
    img_basenames = set()
    for filename in os.listdir(img_folder):
        # 获取文件完整路径
        file_path = os.path.join(img_folder, filename)
        # 跳过文件夹，只处理文件
        if os.path.isfile(file_path):
            # 分离文件名和扩展名
            name, ext = os.path.splitext(filename)
            # 只处理指定格式的图片文件
            if ext.lower() in img_extensions:
                img_basenames.add(name)

    if not img_basenames:
        print("警告：图片文件夹中未找到任何图片文件！")
        return

    print(f"共找到 {len(img_basenames)} 个图片文件")

    # 2. 遍历文本文件夹，删除不匹配的.txt文件
    deleted_count = 0
    for filename in os.listdir(txt_folder):
        file_path = os.path.join(txt_folder, filename)
        # 只处理.txt文件，跳过文件夹
        if os.path.isfile(file_path) and filename.lower().endswith('.txt'):
            # 提取文本文件名（不含扩展名）
            txt_basename = os.path.splitext(filename)[0]

            # 检查是否有对应的图片
            if txt_basename not in img_basenames:
                try:
                    # 删除文件（可选择改为移动到回收站，更安全）
                    # shutil.move(file_path, os.path.join(os.environ['USERPROFILE'], 'Recycle Bin', filename))
                    os.remove(file_path)
                    print(f"已删除无对应图片的文件：{filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"删除文件 {filename} 失败：{str(e)}")

    print(f"\n清理完成！共删除 {deleted_count} 个无对应图片的.txt文件")


def delete_unmatched_img_files(txt_folder, img_folder, img_extensions=None):
    """
    删除图片文件夹中没有对应文本文件的图片文件

    Args:
        txt_folder (str): 文本文件夹路径
        img_folder (str): 图片文件夹路径
        img_extensions (list): 要识别的图片扩展名，默认包含常见格式
    """
    # 默认支持的图片格式（小写）
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # 统一转为小写，避免大小写判断问题
    img_extensions = [ext.lower() for ext in img_extensions]

    # 验证文件夹是否存在
    if not os.path.exists(txt_folder):
        print(f"错误：文本文件夹 {txt_folder} 不存在！")
        return
    if not os.path.exists(img_folder):
        print(f"错误：图片文件夹 {img_folder} 不存在！")
        return

    # 1. 提取所有.txt文件名（不含扩展名）
    txt_basenames = set()
    for filename in os.listdir(txt_folder):
        file_path = os.path.join(txt_folder, filename)
        # 跳过文件夹，只处理.txt文件
        if os.path.isfile(file_path) and filename.lower().endswith('.txt'):
            # 提取文件名（不含扩展名）并加入集合
            txt_basename = os.path.splitext(filename)[0]
            txt_basenames.add(txt_basename.lower())  # 转小写，大小写不敏感匹配

    if not txt_basenames:
        print("警告：文本文件夹中未找到任何.txt文件！")
        return

    print(f"共找到 {len(txt_basenames)} 个.txt文件")

    # 2. 遍历图片文件夹，删除不匹配的图片文件
    deleted_count = 0
    for filename in os.listdir(img_folder):
        file_path = os.path.join(img_folder, filename)
        # 跳过文件夹，只处理文件
        if os.path.isfile(file_path):
            # 分离文件名和扩展名
            img_name, img_ext = os.path.splitext(filename)
            # 检查是否是指定格式的图片
            if img_ext.lower() in img_extensions:
                # 检查是否有对应的.txt文件
                if img_name.lower() not in txt_basenames:
                    try:
                        # 删除无对应txt的图片文件
                        os.remove(file_path)
                        print(f"已删除无对应txt的图片：{filename}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除图片 {filename} 失败：{str(e)}")

    print(f"\n清理完成！共删除 {deleted_count} 个无对应.txt文件的图片")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 请修改为你的实际文件夹路径
    IMAGE_FOLDER = r"./data/test_data/train/images"  # 图片文件夹
    TXT_FOLDER = r"./data/test_data/train/labels"  # 文本文件夹

    # 执行清理
    delete_unmatched_img_files(
        img_folder=IMAGE_FOLDER,
        txt_folder=TXT_FOLDER,
        # 可自定义需要识别的图片格式
        img_extensions=['.jpg', '.png', '.jpeg']
    )