import tensorflow as tf
import logging
from datetime import datetime
import os
import json
import psutil
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import shutil

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


# 配置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# GPU内存配置必须在导入其他TensorFlow模块之前完成
try:
    # 设置GPU内存增长
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.info("GPU内存增长已启用")
except Exception as e:
    logging.error(f"GPU配置错误: {e}")

# 检查是否在Colab环境中
def check_colab():
    try:
        import google.colab
        return True
    except:
        return False

def configure_gpu():
    try:
        IN_COLAB = check_colab()
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            # 显示GPU信息
            if IN_COLAB:
                import subprocess
                gpu_info = subprocess.check_output(['nvidia-smi']).decode()
                logging.info(f"GPU信息:\n{gpu_info}")
            logging.info(f"检测到 {len(physical_devices)} 个GPU设备")
        else:
            logging.warning("未检测到GPU，将使用CPU训练")
    except Exception as e:
        logging.error(f"GPU信息获取错误: {e}")

# 在导入其他TensorFlow模块之前配置GPU
configure_gpu()

from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# 数据增强层
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.GaussianNoise(0.1),
])

class ModelConfig:
    def __init__(self, img_size=160, batch_size=32, epochs=50, initial_lr=0.001):
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_path = f"drive/MyDrive/models/{self.model_name}/checkpoints"
        self.logs_path = f"drive/MyDrive/models/{self.model_name}/logs"

        # 创建必要的目录
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.logs_path, exist_ok=True)

def create_datasets(config, train_dir, test_dir):
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    # 定义 AUTOTUNE
    AUTOTUNE = tf.data.AUTOTUNE

    def preprocess_data(image, label):
        image = tf.cast(image, tf.float32)
        do_augment = tf.random.uniform([], dtype=tf.float32) > 0.5

        def augmented():
            x = data_augmentation(image)
            return preprocess_input(x)

        def not_augmented():
            return tf.cast(preprocess_input(image), tf.float16)

        image = tf.cond(do_augment, augmented, not_augmented)
        return image, label

    def get_default_normal_image_path(directory):
        """获取默认普通菌落图片的路径"""
        # 默认图片位于 dataset/00/normal/default_normal.jpg
        default_path = os.path.join(directory, "00", "normal", "default_normal.jpg")
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"找不到默认图片：{default_path}")
        return default_path

    def load_and_balance_class_data(class_path, default_image_path, is_training=True):
        """加载并平衡单个类别的两种图片数据"""
        micro_path = os.path.join(class_path, "microscope")
        normal_path = os.path.join(class_path, "normal")

        # 获取显微镜图片列表
        micro_files = [os.path.join(micro_path, f) for f in os.listdir(micro_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # 获取普通菌落图片列表
        if os.path.exists(normal_path) and os.listdir(normal_path):
            normal_files = [os.path.join(normal_path, f) for f in os.listdir(normal_path)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if not normal_files:  # 如果没有有效的普通菌落图片
                normal_files = [default_image_path]
                use_default = True
            else:
                use_default = False
        else:
            normal_files = [default_image_path]
            use_default = True

        # 计算需要重复的次数
        num_micro = len(micro_files)

        # 重复普通菌落图片列表以匹配显微镜图片数量
        if use_default:
            normal_files_balanced = [normal_files[0]] * num_micro
        else:
            # 循环重复普通菌落图片
            normal_files_balanced = []
            while len(normal_files_balanced) < num_micro:
                normal_files_balanced.extend(normal_files)
            normal_files_balanced = normal_files_balanced[:num_micro]

        return micro_files, normal_files_balanced

    def create_balanced_datasets(directory, is_training=True):
        all_micro_files = []
        all_normal_files = []
        all_labels = []
        valid_classes = []

        # 获取默认图片路径
        default_image_path = get_default_normal_image_path(directory)
        logging.info(f"使用默认图片：{default_image_path}")

        # 获取所有类别
        class_names = sorted(os.listdir(directory))

        # 处理每个类别
        for label, class_name in enumerate(class_names):
            class_path = os.path.join(directory, class_name)
            if not os.path.isdir(class_path):
                continue

            try:
                micro_files, normal_files = load_and_balance_class_data(
                    class_path, default_image_path, is_training)

                if len(micro_files) > 0:  # 只有当有效文件时才添加该类别
                    # 添加到总列表
                    all_micro_files.extend(micro_files)
                    all_normal_files.extend(normal_files)
                    all_labels.extend([label] * len(micro_files))
                    valid_classes.append(class_name)

            except Exception as e:
                logging.warning(f"处理类别 {class_name} 时出错: {str(e)}")
                continue

        if not all_micro_files:
            raise ValueError("没有找到有效的训练数据！")

        # 创建数据集
        micro_dataset = tf.data.Dataset.from_tensor_slices((all_micro_files, all_labels))
        normal_dataset = tf.data.Dataset.from_tensor_slices((all_normal_files, all_labels))

        # 读取和预处理图片
        def read_image(file_path, label):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
            img = tf.image.resize(img, [config.img_size, config.img_size])
            return img, label

        micro_dataset = micro_dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
        normal_dataset = normal_dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)

        return micro_dataset, normal_dataset, valid_classes

    # 加载训练集
    train_micro, train_normal, class_names = create_balanced_datasets(train_dir, True)

    # 加载测试集
    test_micro, test_normal, _ = create_balanced_datasets(test_dir, False)

    # 计算验证集大小
    total_size = tf.data.experimental.cardinality(train_micro).numpy()
    val_size = int(total_size * 0.2)
    train_size = total_size - val_size

    # 分割训练集和验证集
    train_micro = train_micro.shuffle(total_size, seed=123)
    train_normal = train_normal.shuffle(total_size, seed=123)

    train_micro_final = train_micro.take(train_size)
    val_micro = train_micro.skip(train_size)

    train_normal_final = train_normal.take(train_size)
    val_normal = train_normal.skip(train_size)

    # 预处理数据集
    train_micro_final = (train_micro_final
        .map(preprocess_data, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1000))

    train_normal_final = (train_normal_final
        .map(preprocess_data, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1000))

    val_micro = (val_micro
        .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
             num_parallel_calls=AUTOTUNE)
        .cache())

    val_normal = (val_normal
        .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
             num_parallel_calls=AUTOTUNE)
        .cache())

    test_micro = (test_micro
        .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
             num_parallel_calls=AUTOTUNE)
        .cache())

    test_normal = (test_normal
        .map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y),
             num_parallel_calls=AUTOTUNE)
        .cache())

    # 合并数据集并设置batch_size
    train_dataset = tf.data.Dataset.zip((train_micro_final, train_normal_final)).map(
        lambda m, n: ({'micro_input': m[0], 'normal_input': n[0]}, m[1])
    ).batch(config.batch_size).prefetch(buffer_size=AUTOTUNE)

    val_dataset = tf.data.Dataset.zip((val_micro, val_normal)).map(
        lambda m, n: ({'micro_input': m[0], 'normal_input': n[0]}, m[1])
    ).batch(config.batch_size).prefetch(buffer_size=AUTOTUNE)

    test_dataset = tf.data.Dataset.zip((test_micro, test_normal)).map(
        lambda m, n: ({'micro_input': m[0], 'normal_input': n[0]}, m[1])
    ).batch(config.batch_size).prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, test_dataset, class_names

def build_model(num_classes, config, base_model_name=input('EfficientNetB0, ResNet50, EfficientNetV2, VGG16, InceptionV3, DenseNet121, MobileNetV2')):
    print('================')
    print(f'Using {base_model_name}')
    print('================')
    # 根据名称选择基础模型
    if base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(config.img_size, config.img_size, 3))
    elif base_model_name == 'ResNet50':
        from tensorflow.keras.applications import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(config.img_size, config.img_size, 3))
    elif base_model_name == 'EfficientNetV2':
        from tensorflow.keras.applications import EfficientNetV2B0
        base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(config.img_size, config.img_size, 3))
    elif base_model_name == 'VGG16':
        from tensorflow.keras.applications import VGG16
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(config.img_size, config.img_size, 3))
    elif base_model_name == 'InceptionV3':
        from tensorflow.keras.applications import InceptionV3
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(config.img_size, config.img_size, 3))
    elif base_model_name == 'DenseNet121':
        from tensorflow.keras.applications import DenseNet121
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(config.img_size, config.img_size, 3))
    elif base_model_name == 'MobileNetV2':
        from tensorflow.keras.applications import MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(config.img_size, config.img_size, 3))
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")

    # 构建双路输入模型
    input_micro = layers.Input(shape=(config.img_size, config.img_size, 3), name='micro_input')
    x_micro = base_model(input_micro, training=False)
    x_micro = layers.GlobalAveragePooling2D()(x_micro)
    x_micro = layers.BatchNormalization()(x_micro)
    x_micro = layers.Dense(256, activation='relu')(x_micro)
    x_micro = layers.Dropout(0.5)(x_micro)

    input_normal = layers.Input(shape=(config.img_size, config.img_size, 3), name='normal_input')
    x_normal = base_model(input_normal, training=False)
    x_normal = layers.GlobalAveragePooling2D()(x_normal)
    x_normal = layers.BatchNormalization()(x_normal)
    x_normal = layers.Dense(256, activation='relu')(x_normal)
    x_normal = layers.Dropout(0.5)(x_normal)

    combined = layers.Concatenate()([x_micro, x_normal])
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=[input_micro, input_normal], outputs=outputs)

    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    return model

def get_callbacks(config):
    return [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_path, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        TensorBoard(
            log_dir=config.logs_path,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        CSVLogger(
            os.path.join(config.logs_path, 'training_log.csv'),
            append=True
        )
    ]

def plot_history(histories, labels, save_path):
    plt.figure(figsize=(12, 5))
    for history, label in zip(histories, labels):
        plt.plot(history.history['accuracy'], label=f'{label} - train')
        plt.plot(history.history['val_accuracy'], label=f'{label} - val')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    try:
        # 检查是否在Colab环境中并挂载Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logging.info("Google Drive挂载成功")
        except:
            logging.warning("未在Colab环境中或Drive挂载失败")

        # 配置参数
        config = ModelConfig(
            img_size=160,
            batch_size=32,
            epochs=50,
            initial_lr=0.001
        )

        # 清理内存
        gc.collect()
        tf.keras.backend.clear_session()

        # 创建数据集
        train_dataset, val_dataset, test_dataset, class_names = create_datasets(
            config,
            './dataset',
            './testdata'
        )

        # 保存类别信息
        with open(os.path.join(config.checkpoint_path, 'class_names.json'), 'w') as f:
            json.dump(class_names, f)

        # 构建模型
        model = build_model(len(class_names), config)

        # 编译模型
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.initial_lr,
                epsilon=1e-7,
                beta_1=0.9,
                beta_2=0.999,
                amsgrad=True
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 显示模型信息
        model.summary()

        # 训练模型
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=config.epochs,
            callbacks=get_callbacks(config),
            verbose=1
        )

        # 定期清理内存
        gc.collect()

        # 评估模型
        test_loss, test_accuracy = model.evaluate(
            test_dataset
        )
        logging.info(f"测试集准确率: {test_accuracy:.4f}")

        # 预测和评估
        y_pred = []
        y_true = []

        # 分批处理预测以节省内存
        for batch in test_dataset:
            inputs, labels = batch
            predictions = model.predict(inputs, batch_size=32)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())

            # 清理预测过程中的内存
            del predictions
            gc.collect()

        # 保存评估结果
        print("\n🎯 分类性能报告：")
        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            labels=range(len(class_names)),  # 明确指定标签
            zero_division=0  # 处理零除错误
        )
        print(report)
        logging.info("\n分类报告:\n" + report)

        with open(os.path.join(config.logs_path, 'classification_report.txt'), 'w') as f:
            f.write(report)

        # 计算并保存混淆矩阵
        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=range(len(class_names))  # 明确指定标签
        )

        # 设置中文字体
        plt.style.use('default')

        # 原始混淆矩阵
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('Original Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(config.logs_path, 'confusion_matrix.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()

        # 归一化混淆矩阵
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    annot_kws={"size": 8})
        plt.title('Normalized Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(config.logs_path, 'normalized_confusion_matrix.png'),
                    bbox_inches='tight', dpi=300)
        plt.show()

        # 保存最终模型
        model.save(os.path.join(config.checkpoint_path, 'final_model.keras'))
        logging.info(f"模型已保存到 {config.checkpoint_path}")

    except Exception as e:
        logging.error(f"训练过程中出错: {e}")
        raise
    finally:
        # 清理最终的内存
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == '__main__':
    main()
