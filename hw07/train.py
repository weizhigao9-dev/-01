import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. 基础设置与路径
# 假设你的数据集下载后解压在当前目录的 chest_xray 文件夹下
# 如果使用 Kaggle，请修改为: base_dir = '/kaggle/input/chest-xray-pneumonia/chest_xray'
base_dir = './chest_xray' 
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
figures_dir = './figures'
os.makedirs(figures_dir, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = (150, 150)
EPOCHS = 10

print("=== 1. 数据准备与加载 ===")
# 使用 image_dataset_from_directory 按 8:2 划分训练集和验证集
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False # 测试集不要打乱，方便后续算指标
)

class_names = train_dataset.class_names
print(f"类别名称: {class_names}") # ['NORMAL', 'PNEUMONIA']

# 为了提高性能，使用 prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("\n=== 2. 模型构建 (自定义 CNN) ===")
# 数据增强层
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
])

model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dropout(0.5), # 缓解过拟合
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid') # 二分类
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 计算类别权重 (处理类别不平衡: Pneumonia 远多于 Normal)
# 训练集中 Normal 大约 1341 张，Pneumonia 大约 3875 张 (总数 5216)
weight_for_0 = (1 / 1341) * (5216 / 2.0)
weight_for_1 = (1 / 3875) * (5216 / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}

print("\n=== 3. 模型训练 ===")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weight
)

print("\n=== 4. 模型评估与结果绘制 ===")
# 绘制 Loss 和 Accuracy 曲线
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig(os.path.join(figures_dir, 'training_curves.png'))
plt.show()

# 在测试集上进行预测
y_true = []
for images, labels in test_dataset.unbatch():
    y_true.append(labels.numpy())
y_true = np.array(y_true)

predictions = model.predict(test_dataset)
y_pred = (predictions > 0.5).astype(int).reshape(-1)

# 计算各项指标 (宏观/针对Pneumonia类)
acc_score = accuracy_score(y_true, y_pred)
prec_score = precision_score(y_true, y_pred)
rec_score = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"\n[Test Set Metrics]")
print(f"Accuracy : {acc_score:.4f}")
print(f"Precision: {prec_score:.4f}")
print(f"Recall   : {rec_score:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

# 绘制混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.savefig(os.path.join(figures_dir, 'confusion_matrix.png'))
plt.show()
