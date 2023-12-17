import os

def delete_images(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 遍历文件夹中的每个文件
    for file in files:
        file_path = os.path.join(folder_path, file)

        # 检查文件是否是图片
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 检查文件名是否以1、3、5、7、9开头
            if file.startswith(('1', '3', '5', '7', '9')):
                try:
                    # 删除文件
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# 调用函数并传入文件夹路径
# folder_path = './data/training/'
folder_path = './data/testing/'
delete_images(folder_path)
