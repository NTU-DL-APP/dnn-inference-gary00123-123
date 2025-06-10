import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 固定隨機種子，穩定結果
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 建立模型資料夾
os.makedirs("model", exist_ok=True)

# 載入與預處理資料
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

# 模型架構（最穩組合）
model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(384, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 使用 Adam（比 SGD 更穩）
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 訓練模型（適當的訓練量）
model.fit(
    x_train,
    y_train,
    epochs=25,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# 評估結果
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# 儲存權重
model_path = "model/fashion_mnist"
weights = {}
for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Dense):
        w, b = layer.get_weights()
        weights[f'dense_{i}_kernel:0'] = w
        weights[f'dense_{i}_bias:0'] = b
np.savez(f"{model_path}.npz", **weights)

# 儲存模型架構
arch = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, layers.Dense):
        arch.append({
            "name": f"dense_{i}",
            "type": "Dense",
            "config": {"activation": layer.activation.__name__},
            "weights": [f'dense_{i}_kernel:0', f'dense_{i}_bias:0']
        })

with open(f"{model_path}.json", "w") as f:
    json.dump(arch, f, indent=2)
