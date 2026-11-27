import os
import random
import shutil

# 配置路径
source_root = "./drive/MyDrive/CNN/dataset"  # 显微镜下图片路径
normal_root = "./drive/MyDrive/CNN/normal"   # 普通菌落照片路径
train_root = "./dataset"
test_root = "./testdata"
default_normal = "default_normal.jpg"        # 默认的普通菌落照片

# 支持的图像格式
image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif', '.webp')

# 测试集比例
test_ratio = 0.2  # 每类抽取20%的图片作为测试集

# 清空旧目录
if os.path.exists(train_root):
    shutil.rmtree(train_root)
if os.path.exists(test_root):
    shutil.rmtree(test_root)

os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

# 创建用于存储显微镜图片和普通照片的子目录
def create_subdirs(base_path, class_name):
    micro_path = os.path.join(base_path, class_name, "microscope")
    normal_path = os.path.join(base_path, class_name, "normal")
    os.makedirs(micro_path, exist_ok=True)
    os.makedirs(normal_path, exist_ok=True)
    return micro_path, normal_path

# 遍历每个类别文件夹
for class_name in sorted(os.listdir(source_root)):
    class_path = os.path.join(source_root, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📁 当前正在处理类别：{class_name}")

    # 获取显微镜图片
    micro_images = [f for f in os.listdir(class_path)
                    if f.lower().endswith(image_extensions)]

    if len(micro_images) == 0:
        print(f"⚠️ 警告：{class_name} 类别中未找到显微镜图片")
        continue

    # 获取对应的普通菌落照片
    normal_class_path = os.path.join(normal_root, class_name)
    if os.path.exists(normal_class_path):
        normal_images = [f for f in os.listdir(normal_class_path)
                        if f.lower().endswith(image_extensions)]
    else:
        print(f"⚠️ 警告：{class_name} 类别没有对应的普通菌落照片，将使用默认图片")
        normal_images = [default_normal]

    # 划分训练集和测试集
    random.shuffle(micro_images)
    num_test = max(1, int(len(micro_images) * test_ratio))

    test_micro = micro_images[:num_test]
    train_micro = micro_images[num_test:]

    # 为普通照片创建相同数量的列表（可能重复使用）
    train_normal = normal_images * (len(train_micro) // len(normal_images) + 1)
    test_normal = normal_images * (len(test_micro) // len(normal_images) + 1)
    train_normal = train_normal[:len(train_micro)]
    test_normal = test_normal[:len(test_micro)]

    # 创建目录结构
    train_micro_path, train_normal_path = create_subdirs(train_root, class_name)
    test_micro_path, test_normal_path = create_subdirs(test_root, class_name)

    # 复制训练集图片
    for micro_file, normal_file in zip(train_micro, train_normal):
        shutil.copy(os.path.join(class_path, micro_file),
                   os.path.join(train_micro_path, micro_file))
        if normal_file == default_normal:
            shutil.copy(os.path.join(normal_root, default_normal),
                       os.path.join(train_normal_path, normal_file))
        else:
            shutil.copy(os.path.join(normal_class_path, normal_file),
                       os.path.join(train_normal_path, normal_file))

    # 复制测试集图片
    for micro_file, normal_file in zip(test_micro, test_normal):
        shutil.copy(os.path.join(class_path, micro_file),
                   os.path.join(test_micro_path, micro_file))
        if normal_file == default_normal:
            shutil.copy(os.path.join(normal_root, default_normal),
                       os.path.join(test_normal_path, normal_file))
        else:
            shutil.copy(os.path.join(normal_class_path, normal_file),
                       os.path.join(test_normal_path, normal_file))

print("\n✅ 数据集重构完毕！")
print(f"训练集在 {train_root}，测试集在 {test_root}")
print("每个类别下都包含 microscope 和 normal 两个子文件夹")
